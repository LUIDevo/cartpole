using Godot;
using System;

public partial class Pole : RigidBody2D
{
	private bool _stateInitialized = false;

	private float Rand(float min, float max) => min + (float)DataLog.Rng.NextDouble() * (max - min);

	public override void _Ready()
	{
		Mass = Rand(1.0f, 5.0f);

		var mat = PhysicsMaterialOverride ?? new PhysicsMaterial();
		mat.Friction = Rand(0.1f, 0.5f);
		mat.Bounce   = Rand(0.0f, 0.2f);
		PhysicsMaterialOverride = mat;
	}

	public override void _IntegrateForces(PhysicsDirectBodyState2D state)
	{
		if (_stateInitialized) return;
		_stateInitialized = true;

		float startAngle = Rand(-0.25f, 0.25f);
		state.Transform = new Transform2D(startAngle, state.Transform.Origin);

		state.AngularVelocity = Rand(-1.0f, 1.0f);
		state.LinearVelocity  = new Vector2(Rand(-50f, 50f), 0f);
	}
}
