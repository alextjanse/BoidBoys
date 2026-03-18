

/**
 * KD-Tree node for 2D/3D spatial partitioning
 * Stores references to children and parent via indices for efficient memory layout
 */
export class KDTreeNode {
    constructor(point, depth, nodeIndex) {
        this.point = point;           // [x, y] or [x, y, z]
        this.depth = depth;           // Current splitting axis
        this.nodeIndex = nodeIndex;   // Index in the nodes array
        this.leftIndex = -1;          // Index of left child (-1 if none)
        this.rightIndex = -1;         // Index of right child (-1 if none)
        this.parentIndex = -1;        // Index of parent (-1 if root)
    }
}

/**
 * KD-Tree for 2D spatial queries
 * Maintains both a tree structure and a flat node array for efficient access
 */
export class KDTree2D {
    constructor(points = []) {
        this.nodes = [];              // Flat array of all nodes
        this.rootIndex = -1;          // Index of root node
        this.dimensions = 2;
        
        if (points.length > 0) {
            this.build(points);
        }
    }

    /**
     * Build tree from array of [x, y] points
     */
    build(points) {
        this.nodes = [];
        if (points.length === 0) return;
        
        const indices = points.map((_, i) => i);
        this.rootIndex = this._buildRecursive(points, indices, 0, -1);
    }

    _buildRecursive(points, indices, depth, parentIndex) {
        if (indices.length === 0) return -1;

        const axis = depth % this.dimensions;
        const sorted = indices.sort((a, b) => points[a][axis] - points[b][axis]);
        const median = Math.floor(sorted.length / 2);
        const medianIdx = sorted[median];

        const nodeIndex = this.nodes.length;
        const node = new KDTreeNode(points[medianIdx], depth, nodeIndex);
        node.parentIndex = parentIndex;
        this.nodes.push(node);

        node.leftIndex = this._buildRecursive(
            points,
            sorted.slice(0, median),
            depth + 1,
            nodeIndex
        );
        node.rightIndex = this._buildRecursive(
            points,
            sorted.slice(median + 1),
            depth + 1,
            nodeIndex
        );

        return nodeIndex;
    }

    /**
     * Find nearest neighbor to query point
     */
    nearest(queryPoint) {
        if (this.rootIndex === -1) return null;
        
        let best = { nodeIndex: -1, distance: Infinity };
        this._nearestRecursive(this.rootIndex, queryPoint, best);
        
        return best.nodeIndex !== -1 ? this.nodes[best.nodeIndex] : null;
    }

    _nearestRecursive(nodeIndex, query, best) {
        if (nodeIndex === -1) return;
        
        const node = this.nodes[nodeIndex];
        const distance = this._squaredDistance(node.point, query);

        if (distance < best.distance) {
            best.distance = distance;
            best.nodeIndex = nodeIndex;
        }

        const axis = node.depth % this.dimensions;
        const axisDist = query[axis] - node.point[axis];

        const nearSide = axisDist < 0 ? node.leftIndex : node.rightIndex;
        const farSide = axisDist < 0 ? node.rightIndex : node.leftIndex;

        this._nearestRecursive(nearSide, query, best);

        if (axisDist * axisDist < best.distance) {
            this._nearestRecursive(farSide, query, best);
        }
    }

    /**
     * Find all points within radius from query point
     */
    rangeSearch(queryPoint, radius) {
        const radiusSq = radius * radius;
        const results = [];
        
        if (this.rootIndex !== -1) {
            this._rangeSearchRecursive(this.rootIndex, queryPoint, radiusSq, results);
        }
        
        return results;
    }

    _rangeSearchRecursive(nodeIndex, query, radiusSq, results) {
        if (nodeIndex === -1) return;
        
        const node = this.nodes[nodeIndex];
        const distance = this._squaredDistance(node.point, query);

        if (distance <= radiusSq) {
            results.push(node);
        }

        const axis = node.depth % this.dimensions;
        const axisDist = query[axis] - node.point[axis];

        if (axisDist < 0) {
            this._rangeSearchRecursive(node.leftIndex, query, radiusSq, results);
            if (axisDist * axisDist <= radiusSq) {
                this._rangeSearchRecursive(node.rightIndex, query, radiusSq, results);
            }
        } else {
            this._rangeSearchRecursive(node.rightIndex, query, radiusSq, results);
            if (axisDist * axisDist <= radiusSq) {
                this._rangeSearchRecursive(node.leftIndex, query, radiusSq, results);
            }
        }
    }

    _squaredDistance(p1, p2) {
        let sum = 0;
        for (let i = 0; i < this.dimensions; i++) {
            const diff = p1[i] - p2[i];
            sum += diff * diff;
        }
        return sum;
    }
}

/**
 * KD-Tree for 3D spatial queries
 * Maintains both a tree structure and a flat node array for efficient access
 */
export class KDTree3D {
    constructor(points = []) {
        this.nodes = [];              // Flat array of all nodes
        this.rootIndex = -1;          // Index of root node
        this.dimensions = 3;
        
        if (points.length > 0) {
            this.build(points);
        }
    }

    /**
     * Build tree from array of [x, y, z] points
     */
    build(points) {
        this.nodes = [];
        if (points.length === 0) return;
        
        const indices = points.map((_, i) => i);
        this.rootIndex = this._buildRecursive(points, indices, 0, -1);
    }

    _buildRecursive(points, indices, depth, parentIndex) {
        if (indices.length === 0) return -1;

        const axis = depth % this.dimensions;
        const sorted = indices.sort((a, b) => points[a][axis] - points[b][axis]);
        const median = Math.floor(sorted.length / 2);
        const medianIdx = sorted[median];

        const nodeIndex = this.nodes.length;
        const node = new KDTreeNode(points[medianIdx], depth, nodeIndex);
        node.parentIndex = parentIndex;
        this.nodes.push(node);

        node.leftIndex = this._buildRecursive(
            points,
            sorted.slice(0, median),
            depth + 1,
            nodeIndex
        );
        node.rightIndex = this._buildRecursive(
            points,
            sorted.slice(median + 1),
            depth + 1,
            nodeIndex
        );

        return nodeIndex;
    }

    /**
     * Find nearest neighbor to query point
     */
    nearest(queryPoint) {
        if (this.rootIndex === -1) return null;
        
        let best = { nodeIndex: -1, distance: Infinity };
        this._nearestRecursive(this.rootIndex, queryPoint, best);
        
        return best.nodeIndex !== -1 ? this.nodes[best.nodeIndex] : null;
    }

    _nearestRecursive(nodeIndex, query, best) {
        if (nodeIndex === -1) return;
        
        const node = this.nodes[nodeIndex];
        const distance = this._squaredDistance(node.point, query);

        if (distance < best.distance) {
            best.distance = distance;
            best.nodeIndex = nodeIndex;
        }

        const axis = node.depth % this.dimensions;
        const axisDist = query[axis] - node.point[axis];

        const nearSide = axisDist < 0 ? node.leftIndex : node.rightIndex;
        const farSide = axisDist < 0 ? node.rightIndex : node.leftIndex;

        this._nearestRecursive(nearSide, query, best);

        if (axisDist * axisDist < best.distance) {
            this._nearestRecursive(farSide, query, best);
        }
    }

    /**
     * Find all points within radius from query point
     */
    rangeSearch(queryPoint, radius) {
        const radiusSq = radius * radius;
        const results = [];
        
        if (this.rootIndex !== -1) {
            this._rangeSearchRecursive(this.rootIndex, queryPoint, radiusSq, results);
        }
        
        return results;
    }

    _rangeSearchRecursive(nodeIndex, query, radiusSq, results) {
        if (nodeIndex === -1) return;
        
        const node = this.nodes[nodeIndex];
        const distance = this._squaredDistance(node.point, query);

        if (distance <= radiusSq) {
            results.push(node);
        }

        const axis = node.depth % this.dimensions;
        const axisDist = query[axis] - node.point[axis];

        if (axisDist < 0) {
            this._rangeSearchRecursive(node.leftIndex, query, radiusSq, results);
            if (axisDist * axisDist <= radiusSq) {
                this._rangeSearchRecursive(node.rightIndex, query, radiusSq, results);
            }
        } else {
            this._rangeSearchRecursive(node.rightIndex, query, radiusSq, results);
            if (axisDist * axisDist <= radiusSq) {
                this._rangeSearchRecursive(node.leftIndex, query, radiusSq, results);
            }
        }
    }

    _squaredDistance(p1, p2) {
        let sum = 0;
        for (let i = 0; i < this.dimensions; i++) {
            const diff = p1[i] - p2[i];
            sum += diff * diff;
        }
        return sum;
    }
}