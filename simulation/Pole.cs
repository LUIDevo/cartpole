using Godot;
using System;

public partial class Pole : RigidBody2D
{
	private bool _stateInitialized = false;

	// Uniform random float in [min, max], from the shared seeded RNG.
	private float Rand(float min, float max) => min + (float)DataLog.Rng.NextDouble() * (max - min);

	public override void _Ready()
	{
		// Mass and material are plain properties — safe to set here.
		Mass = Rand(0.5f, 20.0f); // kg (real pole weight)

		var mat = PhysicsMaterialOverride ?? new PhysicsMaterial();
		mat.Friction = Rand(0.0f, 0.8f); // surface friction variation
		mat.Bounce   = Rand(0.0f, 0.5f); // bounciness variation
		PhysicsMaterialOverride = mat;
	}

	// RigidBody2D velocity/rotation must be set through the physics state, not in
	// _Ready() — the physics server overwrites _Ready values on the first frame.
	// Apply the random starting motion exactly once.
	public override void _IntegrateForces(PhysicsDirectBodyState2D state)
	{
		if (_stateInitialized) return;
		_stateInitialized = true;

		// Starting tilt angle (pole rotates about the pin pivot at its origin).
		// Kept mild (~±14°): from ±46° with heavy spin the pole is often unsavable,
		// so episodes carried no learning signal and the policy never improved.
		float startAngle = Rand(-0.25f, 0.25f); // rad, ~±14°
		state.Transform = new Transform2D(startAngle, state.Transform.Origin);

		state.AngularVelocity = Rand(-1.0f, 1.0f);               // rad/s starting spin
		state.LinearVelocity  = new Vector2(Rand(-50f, 50f), 0f);   // starting push
	}
}
