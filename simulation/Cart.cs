using Godot;
using System;

public partial class Cart : CharacterBody2D
{
	// Base properties of the cart itself
	[Export] public float CartMass = 5.0f;       // Mass of the cart (kg)
	[Export] public float PoleMass = 2.0f;       // Mass of the pole (kg) — change this to test!
	
	// Motor limits
	[Export] public float MaxMotorForce = 7000f; // Total force the motor can exert (Newtons)
	[Export] public float MaxMotorPower = 2100f; // Total power limit of the motor (Watts)

	public override void _PhysicsProcess(double delta)
	{
		Vector2 currentVelocity = Velocity;
		Vector2 direction = Input.GetVector("ui_left", "ui_right", "ui_up", "ui_down");

		// 1. Calculate total mass
		float totalMass = CartMass + PoleMass;

		// 2. Dynamically calculate Acceleration using F = ma -> a = F/m
		float dynamicAcceleration = MaxMotorForce / totalMass;

		// 3. Dynamically calculate Max Speed based on power limits (Power = Force * Velocity)
		// If the mass is heavy, the motor can't push it as fast.
		float dynamicMaxSpeed = MaxMotorPower / (totalMass * 0.5f); 
		
		// Hard-cap the max speed so it doesn't become infinite if mass is 0
		dynamicMaxSpeed = Mathf.Min(dynamicMaxSpeed, 500f); 

		// 4. Apply the movement logic using our dynamic values
		if (direction.X != 0) 
		{
			currentVelocity.X = Mathf.MoveToward(
				currentVelocity.X,
				direction.X * dynamicMaxSpeed,
				dynamicAcceleration * (float)delta
			);
		}
		else 
		{
			currentVelocity.X = Mathf.MoveToward(
				currentVelocity.X,
				0,
				dynamicAcceleration * (float)delta
			);
		}

		Velocity = currentVelocity;
		MoveAndSlide();
	}
}
