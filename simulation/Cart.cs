using Godot;
using System;

public partial class Cart : CharacterBody2D
{
	[Export] public float CartMass = 5.0f;
	[Export] public float PoleMass = 2.0f;

	[Export] public float MaxMotorForce = 7000f;
	[Export] public float MaxMotorPower = 2100f;

	private float _motorDeadzone;
	private float _motorExponent;
	private float _motorBias;

	private float _command   = 0f;
	private int   _holdTicks = 0;

	public const float TrackLength      = 1000f;
	public const float HalfTrackLength  = TrackLength / 2f;
	private float _trackCenterX;

	private float Rand(float min, float max) => min + (float)DataLog.Rng.NextDouble() * (max - min);

	public override void _Ready()
	{
		CartMass      = Rand(3.0f, 8.0f);
		PoleMass      = Rand(1.0f, 4.0f);
		MaxMotorForce = Rand(8000f, 16000f);
		MaxMotorPower = Rand(4000f, 7000f);

		_motorDeadzone = Rand(0.0f, 0.05f);
		_motorExponent = Rand(0.9f, 1.2f);
		_motorBias     = Rand(-0.03f, 0.03f);

		Velocity = new Vector2(Rand(-60f, 60f), 0f);

		_trackCenterX = Position.X;
	}

	public float CartOffset() => Position.X - _trackCenterX;

	public float NormalizedCartPos() => CartOffset() / HalfTrackLength;

	public static readonly float FailAngle = Mathf.Pi / 2f;

	public bool CartAtEnd() => Mathf.Abs(CartOffset()) >= HalfTrackLength;

	public bool IsTerminal()
	{
		var (_, _, angle) = Observe();
		return Mathf.Abs(angle) > FailAngle || CartAtEnd();
	}

	public float Reward()
	{
		var (_, angVel, angle) = Observe();
		float posN = CartOffset() / HalfTrackLength;
		return 1.0f - (angle * angle + 0.1f * angVel * angVel + 0.5f * posN * posN);
	}

	private float MotorForce(float u)
	{
		u = Mathf.Clamp(u + _motorBias, -1f, 1f);
		float mag = Mathf.Abs(u);
		if (mag < _motorDeadzone) return 0f;
		mag = (mag - _motorDeadzone) / (1f - _motorDeadzone);
		mag = Mathf.Pow(mag, _motorExponent);
		return Mathf.Sign(u) * mag * MaxMotorForce;
	}

	public bool ExternalControl = false;

	public (float cartVel, float poleAngVel, float poleAngle) Observe()
	{
		var pole = GetNodeOrNull<RigidBody2D>("../Node2D");
		return (Velocity.X, pole?.AngularVelocity ?? 0f, pole?.Rotation ?? 0f);
	}

	public const float MaxCartVel    = 1000f;
	public const float MaxPoleAngVel = 10f;
	public static readonly float MaxPoleAngle = Mathf.Pi;

	private static float WrapAngle(float a) => Mathf.Atan2(Mathf.Sin(a), Mathf.Cos(a));

	public (float cartVel, float poleAngVel, float poleAngle) ObserveNormalized()
	{
		var (v, av, ang) = Observe();
		return (v / MaxCartVel, av / MaxPoleAngVel, WrapAngle(ang) / MaxPoleAngle);
	}

	public void ApplyCommand(float command, double delta)
	{
		float totalMass = CartMass + PoleMass;
		float accel     = MotorForce(command) / totalMass;
		float maxSpeed  = Mathf.Min(MaxMotorPower / (totalMass * 0.5f), 1000f);

		Vector2 v = Velocity;
		v.X = Mathf.Clamp(v.X + accel * (float)delta, -maxSpeed, maxSpeed);
		Velocity = v;
		MoveAndSlide();

		ClampToTrack();
	}

	private void ClampToTrack()
	{
		float min = _trackCenterX - HalfTrackLength;
		float max = _trackCenterX + HalfTrackLength;
		if (Position.X <= min || Position.X >= max)
		{
			Position = new Vector2(Mathf.Clamp(Position.X, min, max), Position.Y);
			Vector2 v = Velocity;
			if ((Position.X <= min && v.X < 0f) || (Position.X >= max && v.X > 0f)) v.X = 0f;
			Velocity = v;
		}
	}

	public override void _PhysicsProcess(double delta)
	{
		if (ExternalControl) return;

		if (_holdTicks <= 0)
		{
			_command   = Rand(-1f, 1f);
			_holdTicks = (int)Rand(1f, 15f);
		}
		_holdTicks--;

		var (cartVel, poleAngVel, poleAngle) = ObserveNormalized();
		bool done = IsTerminal() || DataLog.IsLastStep;
		DataLog.WriteRow(cartVel, poleAngVel, poleAngle, NormalizedCartPos(), _command, Reward(), done);
		ApplyCommand(_command, delta);
	}
}
