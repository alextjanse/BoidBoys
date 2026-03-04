/**
 * ES6 Heap class (binary heap).
 * Default is a min-heap. Pass a comparator `(a,b) => a-b`-style to change ordering.
 * Optional `capacity` preallocates backing storage and keeps `_size` to avoid resizing.
 */
class Heap {
	constructor(compare, capacity) {
		this._cmp = typeof compare === 'function' ? compare : (a, b) => a - b;
		if (capacity !== undefined && capacity !== null) {
			if (!Number.isInteger(capacity) || capacity <= 0) throw new TypeError('capacity must be a positive integer');
			this._capacity = capacity;
			this._data = new Array(capacity);
			this._size = 0;
		} else {
			this._capacity = null;
			this._data = [];
			this._size = 0;
		}
	}

	size() { return this._size; }
	isEmpty() { return this._size === 0; }
	peek() { return this._size > 0 ? this._data[0] : undefined; }

	push(value) {
		if (this._capacity !== null) {
			if (this._size >= this._capacity) throw new RangeError('Heap capacity exceeded');
			this._data[this._size] = value;
			this._siftUp(this._size);
			this._size += 1;
			return this._size;
		}
		// dynamic array
		this._data.push(value);
		this._size = this._data.length;
		this._siftUp(this._size - 1);
		return this._size;
	}

	pop() {
		const n = this._size;
		if (n === 0) return undefined;
		const root = this._data[0];
		if (n === 1) {
			if (this._capacity !== null) {
				this._data[0] = undefined;
				this._size = 0;
			} else {
				this._data.pop();
				this._size = 0;
			}
			return root;
		}
		const last = this._data[n - 1];
		if (this._capacity !== null) {
			this._size -= 1;
			this._data[0] = last;
			this._data[this._size] = undefined;
		} else {
			this._data.pop();
			this._size = this._data.length;
			this._data[0] = last;
		}
		this._siftDown(0);
		return root;
	}

	replace(value) {
		if (this._size === 0) {
			this.push(value);
			return undefined;
		}
		const root = this._data[0];
		this._data[0] = value;
		this._siftDown(0);
		return root;
	}

	clear() {
		if (this._capacity !== null) {
			for (let i = 0; i < this._capacity; i++) this._data[i] = undefined;
			this._size = 0;
		} else {
			this._data.length = 0;
			this._size = 0;
		}
	}

	toArray() {
		return this._data.slice(0, this._size);
	}

	_parent(i) { return ((i + 1) >>> 1) - 1; }
	_left(i) { return (i << 1) + 1; }
	_right(i) { return (i << 1) + 2; }

	_siftUp(i) {
		const data = this._data;
		const cmp = this._cmp;
		const item = data[i];
		while (i > 0) {
			const p = this._parent(i);
			const parent = data[p];
			if (cmp(item, parent) >= 0) break;
			data[i] = parent;
			i = p;
		}
		data[i] = item;
	}

	_siftDown(i) {
		const data = this._data;
		const cmp = this._cmp;
		const n = this._size;
		const item = data[i];
		while (true) {
			const l = this._left(i);
			const r = l + 1;
			if (l >= n) break;
			let child = l;
			if (r < n && cmp(data[r], data[l]) < 0) child = r;
			if (cmp(data[child], item) >= 0) break;
			data[i] = data[child];
			i = child;
		}
		data[i] = item;
	}

	/**
	 * Build a heap from an array. Optional `capacity` may be provided.
	 */
	static heapify(arr, compare, capacity) {
		const h = new Heap(compare, capacity);
		const n = arr.length;
		if (h._capacity !== null) {
			if (n > h._capacity) throw new RangeError('Input array larger than capacity');
			for (let i = 0; i < n; i++) h._data[i] = arr[i];
			h._size = n;
		} else {
			h._data = arr.slice();
			h._size = n;
		}
		if (h._size <= 1) return h;
		for (let i = h._parent(h._size - 1); i >= 0; --i) h._siftDown(i);
		return h;
	}
}

// Expose for Node and browser globals
if (typeof module !== 'undefined' && module.exports) {
	module.exports = Heap;
}
if (typeof window !== 'undefined') {
	window.Heap = Heap;
}

/* Usage examples:
// min-heap (default)
const h = new Heap();
h.push(5); h.push(1); h.push(3);
console.log(h.pop()); // 1

// with fixed capacity (preallocated) - throws when full
const h2 = new Heap(undefined, 100);
h2.push(10);
*/

