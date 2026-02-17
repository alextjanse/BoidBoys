using System.Numerics;

namespace BoidBoys.src
{
    public class SteeringBehaviour
    {
        public Vector3 CalculateTotalVelocity(Boid boid, Boid[] neighbours)
        {
            Vector3 total = new();
            total += CalculateSeperation(boid, neighbours);
            total += CalculateAlignment(boid, neighbours);
            total += CalculateCohesion(boid, neighbours);
            return total;
        }

        /// <summary>
        /// Steer to avoid crowding local flockmates 
        /// </summary>
        public Vector3 CalculateSeperation(Boid boid, Boid[] neighbours)
        {
            Vector3 totalVelocity = new();
            Parallel.For(0, neighbours.Length, i =>
            {
                totalVelocity += boid.position - neighbours[i].position;
            });

            return -totalVelocity;
        }

        /// <summary>
        /// Steer towards the average heading of local flockmates 
        /// </summary>
        public Vector3 CalculateAlignment(Boid boid, Boid[] neighbours)
        {
            Vector3 averageVelocity = new();
            Parallel.For(0, neighbours.Length, i =>
            {
                averageVelocity += neighbours[i].velocity;
            });

            averageVelocity /= neighbours.Length;
            return boid.position - averageVelocity;
        }

        /// <summary>
        /// Steer to move toward the average position of local flockmates 
        /// </summary>
        public Vector3 CalculateCohesion(Boid boid, Boid[] neighbours)
        {
            Vector3 averagePosition = new();
            Parallel.For(0, neighbours.Length, i =>
            {
                averagePosition += neighbours[i].position;
            });

            averagePosition /= neighbours.Length;
            return boid.position - averagePosition;
        }
    }
}