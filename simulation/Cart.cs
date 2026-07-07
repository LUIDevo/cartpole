using Godot;
using System;

public partial class Cart : CharacterBody2D
{
	// Base properties of the cart itself (randomized per episode)
	[Export] public float CartMass = 5.0f;       // Mass of the cart (kg)
	[Export] public float PoleMass = 2.0f;       // Pole mass used by the motor math (kg)

	// Motor limits (randomized per episode)
	[Export] public float MaxMotorForce = 7000f; // Peak force the motor can exert (N)
	[Export] public float MaxMotorPower = 2100f; // Power limit -> caps max speed (W)

	// --- "Unknown motor" response randomization ---
	// The dataset LABEL is the raw command u in [-1, 1]. How that command maps to
	// physical force is randomized each episode so the data spans many real motors.
	private float _motorDeadzone; // |u| below this does nothing
	private float _motorExponent; // response curve: force ∝ sign(u)*|u|^exp
	private float _motorBias;     // small directional asymmetry

	// Current random command and how many ticks left to hold it
	private float _command   = 0f;
	private int   _holdTicks = 0;

	// --- Track ---
	// The cart slides along a 1D track centered on its spawn point. Position is
	// measured as signed offset from that center; the track spans ±HalfTrackLength.
	public const float TrackLength      = 1000f;             // sim units, full span
	public const float HalfTrackLength  = TrackLength / 2f;  // ± limit from center
	private float _trackCenterX;                             // spawn X, captured in _Ready

	// Uniform random float in [min, max], from the shared seeded RNG.
	private float Rand(float min, float max) => min + (float)DataLog.Rng.NextDouble() * (max - min);

	public override void _Ready()
	{
		// Cart weight, max accel (force) and max speed (power)
		CartMass      = Rand(2.0f, 12.0f);
		PoleMass      = Rand(0.5f, 6.0f);
		MaxMotorForce = Rand(3000f, 12000f);
		MaxMotorPower = Rand(1000f, 3500f);

		// Randomized motor transfer function
		_motorDeadzone = Rand(0.0f, 0.15f);
		_motorExponent = Rand(0.7f, 1.6f);
		_motorBias     = Rand(-0.1f, 0.1f);

		// Small random starting drift
		Velocity = new Vector2(Rand(-120f, 120f), 0f);

		// Track center = spawn point. Offset is measured from here.
		_trackCenterX = Position.X;
	}

	// Signed cart offset from track center (sim units, raw). Used by reward/done.
	public float CartOffset() => Position.X - _trackCenterX;

	// Normalized cart position feature: |offset| / half-track  ->  [0, 1].
	// NOTE: magnitude only (drops left/right sign), per the requested formula.
	public float NormalizedCartPos() => Mathf.Abs(CartOffset()) / HalfTrackLength;

	// Map raw command u in [-1,1] to a physical force via the randomized motor model.
	private float MotorForce(float u)
	{
		u = Mathf.Clamp(u + _motorBias, -1f, 1f);
		float mag = Mathf.Abs(u);
		if (mag < _motorDeadzone) return 0f;
		mag = (mag - _motorDeadzone) / (1f - _motorDeadzone); // rescale past deadzone
		mag = Mathf.Pow(mag, _motorExponent);                 // nonlinear response
		return Mathf.Sign(u) * mag * MaxMotorForce;
	}

	// When true the cart is driven externally (SimController feeds commands from the
	// control stream); the random self-driving policy below is disabled.
	public bool ExternalControl = false;

	// Current observation: (cart velocity, pole angular velocity, pole angle).
	// Raw physical units — used for episode-end/threshold logic, NOT model-facing.
	public (float cartVel, float poleAngVel, float poleAngle) Observe()
	{
		var pole = GetNodeOrNull<RigidBody2D>("../Node2D");
		return (Velocity.X, pole?.AngularVelocity ?? 0f, pole?.Rotation ?? 0f);
	}

	// --- Observation normalization (model-facing) ---
	// Raw obs are mapped to ~[-1,1] so the network sees a consistent scale. The
	// SAME scales are applied to the training CSV and the live control stream, so a
	// model trained on the data receives identically-scaled inputs at inference.
	public const float MaxCartVel    = 500f;      // matches the ApplyCommand speed clamp
	public const float MaxPoleAngVel = 10f;       // rad/s, headroom over the ±3 start spin
	public static readonly float MaxPoleAngle = Mathf.Pi; // rad, ±180° -> ±1

	// Wrap a continuous rotation into [-π, π] before scaling (the pole can spin past
	// ±180°; without wrapping the normalized angle would blow past ±1).
	private static float WrapAngle(float a) => Mathf.Atan2(Mathf.Sin(a), Mathf.Cos(a));

	// Observation normalized to ~[-1,1] for training data + control commands.
	public (float cartVel, float poleAngVel, float poleAngle) ObserveNormalized()
	{
		var (v, av, ang) = Observe();
		return (v / MaxCartVel, av / MaxPoleAngVel, WrapAngle(ang) / MaxPoleAngle);
	}

	// Apply one motor command (raw u in [-1,1]) for one physics step.
	public void ApplyCommand(float command, double delta)
	{
		float totalMass = CartMass + PoleMass;
		float accel     = MotorForce(command) / totalMass;
		float maxSpeed  = Mathf.Min(MaxMotorPower / (totalMass * 0.5f), 500f);

		Vector2 v = Velocity;
		v.X = Mathf.Clamp(v.X + accel * (float)delta, -maxSpeed, maxSpeed);
		Velocity = v;
		MoveAndSlide();
	}

	public override void _PhysicsProcess(double delta)
	{
		if (ExternalControl) return; // driven by SimController.ApplyCommand()

		// Random policy: pick a new command occasionally, hold it a few ticks
		// (like discrete inputs a driver/agent would send, not per-frame noise).
		if (_holdTicks <= 0)
		{
			_command   = Rand(-1f, 1f);
			_holdTicks = (int)Rand(1f, 15f);
		}
		_holdTicks--;

		var (cartVel, poleAngVel, poleAngle) = ObserveNormalized();
		DataLog.WriteRow(cartVel, poleAngVel, poleAngle, NormalizedCartPos(), _command); // normalized features + label
		ApplyCommand(_command, delta);
	}
}
