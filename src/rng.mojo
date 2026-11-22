struct Rng(Copyable, Movable):
    """Deterministic Xoshiro256** PRNG seeded via SplitMix64."""

    var s0: UInt64
    var s1: UInt64
    var s2: UInt64
    var s3: UInt64

    def __init__(out self, seed: Int):
        var z = UInt64(seed)
        if seed == 0:
            z = UInt64(0x9E3779B97F4A7C15)

        def splitmix(mut state: UInt64) -> UInt64:
            state += UInt64(0x9E3779B97F4A7C15)
            var r = state
            r = (r ^ (r >> 30)) * UInt64(0xBF58476D1CE4E5B9)
            r = (r ^ (r >> 27)) * UInt64(0x94D049BB133111EB)
            return r ^ (r >> 31)

        self.s0 = splitmix(z)
        self.s1 = splitmix(z)
        self.s2 = splitmix(z)
        self.s3 = splitmix(z)

    def next_u64(mut self) -> UInt64:
        var x = self.s1 * UInt64(5)
        var rot = (x << 7) | (x >> 57)
        var result = rot * UInt64(9)

        var t = self.s1 << 17
        self.s2 ^= self.s0
        self.s3 ^= self.s1
        self.s1 ^= self.s2
        self.s0 ^= self.s3
        self.s2 ^= t
        self.s3 = (self.s3 << 45) | (self.s3 >> 19)
        return result

    def random(mut self) -> Float64:
        """Return uniform Float64 in [0.0, 1.0)."""
        comptime INV_2_53: Float64 = 1.0 / 9007199254740992.0
        var bits = self.next_u64() >> 11
        return Float64(bits) * INV_2_53

    def uniform(mut self, lo: Float64, hi: Float64) -> Float64:
        return lo + (hi - lo) * self.random()

    def randint(mut self, n: Int) -> Int:
        """Return integer in [0, n)."""
        if n <= 1:
            return 0
        var idx = Int(self.random() * Float64(n))
        if idx >= n:
            return n - 1
        return idx

    def shuffle_ints(mut self, mut arr: List[Int]):
        var n = len(arr)
        for i in range(n - 1, 0, -1):
            var j = self.randint(i + 1)
            if i != j:
                var tmp = arr[i]
                arr[i] = arr[j]
                arr[j] = tmp
