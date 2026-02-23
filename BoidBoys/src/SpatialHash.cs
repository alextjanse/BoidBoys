using System.Numerics;

namespace BoidBoys.src
{
    /// <summary>
    /// https://matthias-research.github.io/pages/tenMinutePhysics/11-hashing.pdf
    /// </summary>
    public class SpatialHash
    {
        public int[] cellStart;
        public int[] cellEntries;

        public int tableSize;
        public float spacing;

        public SpatialHash(float spacing, int maxNumObjects)
        {
            tableSize = 2 * maxNumObjects;
            this.spacing = spacing;
            cellStart = new int[tableSize + 1];
            cellEntries = new int[maxNumObjects];
        }

        public int HashCoords(Vector3 coord)
        {
            return HashCoords(IntCoord(coord));
        }

        public int HashCoords(Vector3Int intCoord)
        {
            int h = (intCoord.X * 92837111) ^ (intCoord.Y * 689287499) ^ (intCoord.Z * 283923481);
            return Math.Abs(h) % tableSize;
        }

        public Vector3Int IntCoord(Vector3 coord)
        {
            return new Vector3Int(IntCoord(coord.X), IntCoord(coord.Y), IntCoord(coord.Z));
        }

        private int IntCoord(float coord)
        {
            return (int)MathF.Floor(coord / spacing);
        }

        public void Build(Vector3[] pos)
        {
            int numObjects = Math.Min(pos.Length, cellEntries.Length);
            Array.Clear(cellStart);

            // Determine cell sizes
            for (int i = 0; i < numObjects; i++)
            {
                int h = HashCoords(pos[i]);
                cellStart[h]++;
            }

            // Determine cells starts
            int start = 0;
            for (int i = 0; i < tableSize; i++)
            {
                start += cellStart[i];
                cellStart[i] = start;
            }
            cellStart[tableSize] = start;

            // Fill in objects ids
            for (int i = 0; i < numObjects; i++)
            {
                int h = HashCoords(pos[i]);
                cellStart[h]--;
                cellEntries[cellStart[h]] = i;
            }
        }

        public List<int> Query(Vector3[] positions, Vector3 pos, float maxDist)
        {
            var min = IntCoord(pos - new Vector3(maxDist));
            var max = IntCoord(pos + new Vector3(maxDist));

            float maxDistSq = maxDist * maxDist;
            List<int> results = [];

            for (int xi = min.X; xi <= max.X; xi++)
            {
                for (int yi = min.Y; yi <= max.Y; yi++)
                {
                    for (int zi = min.Z; zi <= max.Z; zi++)
                    {
                        int h = HashCoords(new Vector3Int(xi, yi, zi));
                        int start = cellStart[h];
                        int end = cellStart[h + 1];

                        for (int i = start; i < end; i++)
                        {
                            int id = cellEntries[i];

                            if (Vector3.DistanceSquared(positions[id], pos) <= maxDistSq)
                                results.Add(id);
                        }
                    }
                }
            }
            return results;
        }

        public class Vector3Int(int X, int Y, int Z)
        {
            public int X = X;
            public int Y = Y;
            public int Z = Z;
        }
    }
}