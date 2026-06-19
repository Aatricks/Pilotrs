//! Pilot input: keyboard and/or gamepad unified into a single [`StickInput`].
//!
//! Both work at once. The keyboard is digital (a key is held or not), so its
//! axes are run through a rate limiter to feel like a spring-centred stick; a
//! gamepad, if connected, contributes its analog axes on top. Throttle is a
//! persistent level (a real throttle lever doesn't spring back).
//!
//! Default bindings:
//! ```text
//!   pitch    W / ↑  (nose up)      S / ↓  (nose down)
//!   roll     D / →  (right)        A / ←  (left)
//!   yaw      E      (right)        Q      (left)
//!   throttle Shift  (up)           Ctrl   (down)
//!   F  toggle fly-by-wire (FCS) on/off       R  reset to trim
//!   gamepad: left stick = pitch/roll, right stick = yaw/throttle,
//!            A = toggle FCS, Start = reset
//! ```

use std::collections::HashSet;

use fsim_sim::StickInput;
use gilrs::{Axis, Button, EventType, Gilrs};
use three_d::{Event, Key, Modifiers};

/// One frame of pilot intent: the stick plus edge-triggered actions.
#[derive(Debug, Clone, Copy)]
pub struct PilotInput {
    pub stick: StickInput,
    /// The FCS on/off toggle was pressed this frame.
    pub toggle_fbw: bool,
    /// Reset-to-trim was pressed this frame.
    pub reset: bool,
}

/// How fast a held key drives its axis to full deflection (and recenters when
/// released), per second. ~0.3 s to the stop — crisp but not instant.
const KB_RATE: f32 = 3.0;
/// Throttle change rate while Shift/Ctrl is held, per second.
const THROTTLE_RATE: f32 = 0.6;
/// Gamepad stick deadzone.
const DEADZONE: f32 = 0.08;

/// Reads keyboard + gamepad each frame into a [`PilotInput`]. Construct once and
/// feed it the window's events, then [`poll`](Self::poll) it per frame.
pub struct StickSource {
    held: HashSet<Key>,
    mods: Modifiers,
    // Rate-limited keyboard axes (the "stick").
    kb_pitch: f32,
    kb_roll: f32,
    kb_yaw: f32,
    /// Persistent throttle level [0, 1].
    throttle: f32,
    // Edge-triggered actions captured from key/button presses.
    toggle_pending: bool,
    reset_pending: bool,
    // Gamepad (absent if no backend / no pad). Pumped each poll.
    gilrs: Option<Gilrs>,
}

impl StickSource {
    /// Create the input reader, seeding the throttle at `throttle0` (e.g. the
    /// trim throttle, so the aircraft holds speed before the pilot touches it).
    pub fn new(throttle0: f32) -> Self {
        Self {
            held: HashSet::new(),
            mods: Modifiers::default(),
            kb_pitch: 0.0,
            kb_roll: 0.0,
            kb_yaw: 0.0,
            throttle: throttle0.clamp(0.0, 1.0),
            toggle_pending: false,
            reset_pending: false,
            // `Gilrs::new` can fail without a backend; degrade to keyboard-only.
            gilrs: Gilrs::new().ok(),
        }
    }

    /// True if a gamepad is connected (for the HUD hint).
    pub fn has_gamepad(&self) -> bool {
        self.gilrs
            .as_ref()
            .map(|g| g.gamepads().any(|(_, gp)| gp.is_connected()))
            .unwrap_or(false)
    }

    /// Fold this frame's window events into the held-key set and modifiers, and
    /// capture keyboard edges (F = toggle FCS, R = reset). Press edges fire once
    /// per physical key-down (auto-repeat is ignored via the held set).
    pub fn handle_window_events(&mut self, events: &[Event]) {
        for ev in events {
            match ev {
                Event::KeyPress {
                    kind, modifiers, ..
                } => {
                    self.mods = *modifiers;
                    if self.held.insert(*kind) {
                        // fresh press (not auto-repeat)
                        match kind {
                            Key::F => self.toggle_pending = true,
                            Key::R => self.reset_pending = true,
                            _ => {}
                        }
                    }
                }
                Event::KeyRelease {
                    kind, modifiers, ..
                } => {
                    self.mods = *modifiers;
                    self.held.remove(kind);
                }
                Event::ModifiersChange { modifiers } => self.mods = *modifiers,
                _ => {}
            }
        }
    }

    /// Advance the rate-limited keyboard axes and read the gamepad, returning the
    /// combined stick and any edge-triggered actions (consumed by this call).
    pub fn poll(&mut self, dt: f32) -> PilotInput {
        let dt = dt.clamp(0.0, 0.1); // guard a hitched frame
        let down = |k1: Key, k2: Key| self.held.contains(&k1) || self.held.contains(&k2);

        // Keyboard targets from held keys, then slew toward them.
        let p_tgt = bool_axis(down(Key::W, Key::ArrowUp), down(Key::S, Key::ArrowDown));
        let r_tgt = bool_axis(down(Key::D, Key::ArrowRight), down(Key::A, Key::ArrowLeft));
        let y_tgt = bool_axis(self.held.contains(&Key::E), self.held.contains(&Key::Q));
        self.kb_pitch = slew(self.kb_pitch, p_tgt, KB_RATE * dt);
        self.kb_roll = slew(self.kb_roll, r_tgt, KB_RATE * dt);
        self.kb_yaw = slew(self.kb_yaw, y_tgt, KB_RATE * dt);

        // Keyboard throttle (Shift up / Ctrl down).
        if self.mods.shift {
            self.throttle += THROTTLE_RATE * dt;
        }
        if self.mods.ctrl {
            self.throttle -= THROTTLE_RATE * dt;
        }

        // Gamepad: pump the event queue (also catches button edges), then read
        // the analog axes of the first connected pad.
        let (mut pad_pitch, mut pad_roll, mut pad_yaw) = (0.0, 0.0, 0.0);
        let mut pad_throttle: Option<f32> = None;
        if let Some(g) = &mut self.gilrs {
            while let Some(ev) = g.next_event() {
                if let EventType::ButtonPressed(btn, _) = ev.event {
                    match btn {
                        Button::South => self.toggle_pending = true,
                        Button::Start => self.reset_pending = true,
                        _ => {}
                    }
                }
            }
            if let Some((_, gp)) = g.gamepads().find(|(_, gp)| gp.is_connected()) {
                pad_roll = deadzone(gp.value(Axis::LeftStickX));
                pad_pitch = deadzone(gp.value(Axis::LeftStickY)); // up = +1 = nose up
                pad_yaw = deadzone(gp.value(Axis::RightStickX));
                let ry = gp.value(Axis::RightStickY);
                if ry.abs() > DEADZONE {
                    pad_throttle = Some((ry * 0.5 + 0.5).clamp(0.0, 1.0));
                }
            }
        }
        if let Some(t) = pad_throttle {
            // Slew toward the stick position (don't snap) so it matches the
            // rate-limited keyboard feel and a resting-off-center stick can't
            // jump the throttle off its trim seed.
            self.throttle = slew(self.throttle, t, THROTTLE_RATE * dt);
        }
        self.throttle = self.throttle.clamp(0.0, 1.0);

        let stick = StickInput {
            pitch: (self.kb_pitch + pad_pitch).clamp(-1.0, 1.0) as f64,
            roll: (self.kb_roll + pad_roll).clamp(-1.0, 1.0) as f64,
            yaw: (self.kb_yaw + pad_yaw).clamp(-1.0, 1.0) as f64,
            throttle: self.throttle as f64,
        };
        PilotInput {
            stick,
            toggle_fbw: std::mem::take(&mut self.toggle_pending),
            reset: std::mem::take(&mut self.reset_pending),
        }
    }

    /// Reset the stick/throttle to a neutral hold at `throttle0` (e.g. on a
    /// reset-to-trim), without dropping a connected gamepad.
    pub fn recenter(&mut self, throttle0: f32) {
        self.kb_pitch = 0.0;
        self.kb_roll = 0.0;
        self.kb_yaw = 0.0;
        self.throttle = throttle0.clamp(0.0, 1.0);
    }
}

/// +1 if `pos` held, −1 if `neg` held, 0 if neither/both.
fn bool_axis(pos: bool, neg: bool) -> f32 {
    f32::from(pos) - f32::from(neg)
}

/// Move `cur` toward `tgt` by at most `step`.
fn slew(cur: f32, tgt: f32, step: f32) -> f32 {
    let d = tgt - cur;
    if d.abs() <= step {
        tgt
    } else {
        cur + step * d.signum()
    }
}

/// Zero out small stick values inside the deadzone.
fn deadzone(v: f32) -> f32 {
    if v.abs() < DEADZONE {
        0.0
    } else {
        v
    }
}
