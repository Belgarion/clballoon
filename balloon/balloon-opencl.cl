#pragma OPENCL EXTENSION cl_amd_printf : enable

typedef uchar uint8_t;
typedef uint uint32_t;
typedef ulong uint64_t;
typedef char int8_t;
typedef int int32_t;
typedef long int64_t;
typedef uint8_t __sha256_block_t[64];
typedef uint32_t __sha256_hash_t[8];

struct hash_state_lite {
  uint64_t counter;
  uint64_t n_blocks;
  bool has_mixed;
  __global uint8_t *buffer;
  const struct balloon_options *opts;
};

void memcpy(char *dest, const char *source, uint64_t length) {
	for (long i = 0; i < length; i++) {
		dest[i] = source[i];
	}
}

void memcpy32(uint32_t *dest, const uint32_t *source, uint64_t length) {
	for (long i = 0; i < length; i++) {
		dest[i] = source[i];
	}
}

void memset(char *dest, const char ch, uint64_t length) {
	for (long i = 0; i < length; i++) {
		dest[i] = ch;
	}
}

#define SALT_LEN (32)

void cuda_compress (uint64_t *counter, __global uint8_t *out, __global const uint8_t *blocks[], size_t blocks_to_comp);
void cuda_hash_state_mix (struct hash_state_lite *s, int32_t mixrounds, __global uint64_t *prebuf_le);
void cuda_hash_state_fill (struct hash_state_lite *s, const uint8_t *in, size_t inlen, int32_t t_cost, int64_t s_cost);
void device_sha256_168byte(uint8_t *data, __global uint8_t *outhash);
void device_sha256_generic(uint8_t *data, __global uint8_t *outhash, uint32_t len);
void device_sha256_osol(const __sha256_block_t blk, __sha256_hash_t ctx);

#define __device__
__constant const uint32_t __sha256_init[] = {
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
};

#define PREBUF_LEN 409600
#define BLOCK_SIZE (32)

/*uint64_t *device_prebuf_le[20];
uint32_t *device_winning_nonce[20];
uint8_t *device_sbuf[20];
struct hash_state *device_s[20];
uint32_t *device_target[20];
uint32_t *device_is_winning[20];
uint8_t *device_out[20];
uint8_t *device_input[20];
uint8_t *device_sbufs[20];
*/

__global void * block_index(const struct hash_state_lite *s, size_t i) {
	return s->buffer + (BLOCK_SIZE * i);
}
__global void * block_last(const struct hash_state_lite *s) {
	return block_index(s, s->n_blocks - 1);
}
__device__ void cuda_hash_state_extract(const struct hash_state_lite *s, uint8_t out[BLOCK_SIZE]) {
	__global uint8_t *b = (__global uint8_t*)block_last(s);
	//memcpy((char *)out, (const char *)b, BLOCK_SIZE);
	for (int i = 0; i < BLOCK_SIZE; out[i] = b[i], i++);
}

#define CUDA_OUTPUT
//#define DEBUG
__kernel void cudaized_multi(__global uint8_t *hs, int32_t mixrounds,
		__global uint64_t *prebuf_le, __global uint8_t *input, uint32_t len,
		__global uint8_t *output, int64_t s_cost, uint32_t max_nonce,
		int gpuid, __global uint32_t *winning_nonce, uint32_t num_threads, 
		__global uint32_t *device_target, __global uint32_t *is_winning,
		uint32_t num_blocks, __global uint8_t *sbufs) {
	uint32_t id = get_global_id(0);
	uint32_t nonce = (((uint32_t)(input[76]) << 24) | ((uint32_t)(input[77]) << 16) | ((uint32_t)(input[78]) << 8) | (uint32_t)(input[79])) + id;
#ifdef DEBUG
	if (id % 100 == 0) {
	printf("[device] id: %x, prebuf_le[0]: %08x, max_nonce: %x\n", id, prebuf_le[0], max_nonce);
	printf("[device] s_cost: %d, t_cost: %d\n", s_cost, mixrounds);
	printf("[device] input[76-79]: %02x %02x %02x %02x, nonce: %x\n", input[76], input[77], input[78], input[79], nonce);
	}
#endif

	if (nonce > max_nonce || *is_winning) {
#ifdef DEBUG
		printf("[device] early abort, nonce: %x > %x || is_winning(%x)\n", nonce, max_nonce, *is_winning);
#endif
		return;
	}
#ifdef TEST_SHA
	uint8_t data[64];
	for (int i = 0; i < 64; data[i] = 0, i++);
	device_sha256_generic(data, output, 0);
	printf("sha256 null: %02x %02x %02x %02x\n", output[0], output[1], output[2], output[3]);
	data[0] = 'a';
	data[1] = 'b';
	data[2] = 'c';
	device_sha256_generic(data, output, 3);
	printf("sha256 abc: %02x %02x %02x %02x\n", output[0], output[1], output[2], output[3]);
#endif

	uint8_t local_input[80];
#ifdef CUDA_OUTPUT
	uint8_t local_output[32];
#endif
	struct hash_state_lite local_s;
	for (int i = 0; i < len; local_input[i] = input[i], i++);

	__global uint8_t *local_sbuf = hs+4096*BLOCK_SIZE*id;
	//for (int i = 0; i < 4096*BLOCK_SIZE; i++) {
	//	local_sbuf[i] = sbufs[i];
	//}

	local_s.buffer = local_sbuf;
	local_s.n_blocks = 4096;
	((uint32_t*)local_input)[19] = ((nonce & 0xff000000) >> 24) | ((nonce & 0xff0000) >> 8) | ((nonce & 0xff00) << 8) | ((nonce & 0xff) << 24);
	local_s.counter = 0;
	cuda_hash_state_fill(&local_s, local_input, len, mixrounds, s_cost);
	cuda_hash_state_mix (&local_s, mixrounds, prebuf_le);
#ifdef CUDA_OUTPUT
	cuda_hash_state_extract (&local_s, local_output);
	//printf("[device] id: %d\n", id);
	if (id == 0) {
		for (int i = 0; i < 32; output[i] = local_output[i], i++);
		//printf("[device] output[0-3]: %02x %02x %02x %02x\n", output[0], output[1], output[2], output[3]);
	}
	if (((uint32_t*)local_output)[7] < device_target[7]) {
#else
	if (((uint32_t*)(local_sbuf+(4095<<5)))[7] < device_target[7]) {
#endif
		// Assume winning nonce
#ifdef DEBUG
		printf("[Device %d] Winning nonce: %u\n", gpuid, nonce);
#endif
		printf("[Device %d] Winning nonce: %u\n", gpuid, nonce);
		*winning_nonce = nonce;
		*is_winning = 1;
#ifdef CUDA_OUTPUT
		//memcpy((char*)output, (const char*)local_output, 32);
		for (int i = 0; i < 32; output[i] = local_output[i], i++);
#endif
		//__threadfence_system();
		//asm("exit;");
	}
#ifdef DEBUG_CUDA
	printf("[Device %d] leaving cuda\n", gpuid);
#endif
}

__device__ void cuda_expand (uint64_t *counter, __global uint8_t *buf, size_t blocks_in_buf) {
  __global const uint8_t *blocks[1] = { buf };
  __global uint8_t *cur = buf + BLOCK_SIZE;
  for (size_t i = 1; i < blocks_in_buf; i++) {
    cuda_compress (counter, cur, blocks, 1);
    *blocks += BLOCK_SIZE;
    cur += BLOCK_SIZE;
  }
}

__device__ void cuda_compress (uint64_t *counter, __global uint8_t *out, __global const uint8_t *blocks[], size_t blocks_to_comp) {
	uint8_t data[168];
	uint8_t *dp = (uint8_t*)data;
	uint8_t len = BLOCK_SIZE * blocks_to_comp + 8;
	memcpy((char*)dp, (const char*)counter, 8);
	dp += 8;
	for (unsigned int i = 0; i < blocks_to_comp; i++) {
		//memcpy((char*)dp, (const char*)*(blocks+i), BLOCK_SIZE);
		for (int j = 0; j < BLOCK_SIZE; dp[j] = blocks[i][j], j++);
		dp += BLOCK_SIZE;
	}
	device_sha256_generic(data, out, len);
	*counter += 1;
}

__device__ void cuda_hash_state_fill (struct hash_state_lite *s, const uint8_t *in, size_t inlen, int32_t t_cost, int64_t s_cost) {
  uint8_t data[132];
  //uint32_t shalen = 8+SALT_LEN+inlen+8+4;
  uint8_t *dp = (uint8_t*)data;
#ifdef DEBUG
  if (inlen != 80) {
	  printf("inlen != 128 (inlen = %d)!!\n", inlen);
	  if (inlen > 80) inlen = 80;
  }
#endif
  memcpy((char*)dp, (const char*)&s->counter, 8);
  dp += 8;
  memcpy((char*)dp, (const char*)in, SALT_LEN);
  dp += SALT_LEN;
  memcpy((char*)dp, (const char*)in, inlen);
  dp += inlen;
  memcpy((char*)dp, (const char*)&s_cost, 8);
  dp += 8;
  memcpy((char*)dp, (const char*)&t_cost, 4);

  device_sha256_generic(data, s->buffer, 132);
  s->counter++;
  cuda_expand (&s->counter, s->buffer, s->n_blocks);
}




__device__ void cuda_hash_state_mix (struct hash_state_lite *s, int32_t mixrounds, __global uint64_t *prebuf_le) {
	__global uint64_t *buf = prebuf_le;
	__global uint8_t *sbuf = s->buffer;

	//int32_t n_blocks = s->n_blocks;
	const int32_t n_blocks = 4096;
	mixrounds = 4;
	__global uint8_t *last_block = (sbuf + (BLOCK_SIZE*(n_blocks-1)));
	__global uint8_t *blocks[5];
	unsigned char data[8 + BLOCK_SIZE * 5];
	unsigned char *db1 = data + 8;
	unsigned char *db2 = data + 40;
	unsigned char *db3 = data + 72;
	unsigned char *db4 = data + 104;
	unsigned char *db5 = data + 136;
	for (int32_t rounds=0; rounds < mixrounds; rounds++) {
		{ // i = 0
			blocks[0] = last_block;
			blocks[1] = sbuf;
			/*blocks[2] = (sbuf + (BLOCK_SIZE * (*(buf++))));
			blocks[3] = (sbuf + (BLOCK_SIZE * (*(buf++))));
			blocks[4] = (sbuf + (BLOCK_SIZE * (*(buf++))));*/

			blocks[2] = (sbuf + ((*(buf++))));
			blocks[3] = (sbuf + ((*(buf++))));
			blocks[4] = (sbuf + ((*(buf++))));

			// New sha256
			//block = (uint8_t**)blocks;
			memcpy((char*)data, (const char*)&s->counter, 8);
			for (int j = 0; j < BLOCK_SIZE; db1[j] = blocks[0][j], j++);
			for (int j = 0; j < BLOCK_SIZE; db2[j] = blocks[1][j], j++);
			for (int j = 0; j < BLOCK_SIZE; db3[j] = blocks[2][j], j++);
			for (int j = 0; j < BLOCK_SIZE; db4[j] = blocks[3][j], j++);
			for (int j = 0; j < BLOCK_SIZE; db5[j] = blocks[4][j], j++);
			device_sha256_168byte(data, (__global uint8_t*)blocks[1]);
			s->counter++;
		}
		for (size_t i = 1; i < n_blocks; i++) {
			blocks[0] = blocks[1];
			blocks[1] += BLOCK_SIZE;
			/*blocks[2] = (sbuf + (BLOCK_SIZE * (*(buf++))));
			blocks[3] = (sbuf + (BLOCK_SIZE * (*(buf++))));
			blocks[4] = (sbuf + (BLOCK_SIZE * (*(buf++))));*/

			blocks[2] = (sbuf + ((*(buf++))));
			blocks[3] = (sbuf + ((*(buf++))));
			blocks[4] = (sbuf + ((*(buf++))));

			// New sha256
			memcpy((char*)data, (const char*)&s->counter, 8);
			for (int j = 0; j < BLOCK_SIZE; db1[j] = blocks[0][j], j++);
			for (int j = 0; j < BLOCK_SIZE; db2[j] = blocks[1][j], j++);
			for (int j = 0; j < BLOCK_SIZE; db3[j] = blocks[2][j], j++);
			for (int j = 0; j < BLOCK_SIZE; db4[j] = blocks[3][j], j++);
			for (int j = 0; j < BLOCK_SIZE; db5[j] = blocks[4][j], j++);
			device_sha256_168byte(data, (__global uint8_t*)blocks[1]);
			s->counter++;
		}
		//s->has_mixed = true;
	}
#ifdef DEBUG_CUDA
	if (buf - prebuf_le > 49152) printf("prebuf_le max used: %d, mixrounds = %d, n_blocks = %d\n", buf - prebuf_le, mixrounds, n_blocks);
#endif
}

__device__ void device_sha256_168byte(uint8_t *data, __global uint8_t *outhash) {
	// outhash should be 32 byte
	//
	// l = 168byte => 1344bit (requires 3 blocks)
	// (k + 1 + l) mod 512 = 448
	// 512 * 3 = 1536 >= 1344:
	// k = 3*512 - 65 - l = 1536 - 65 - 1344 = 127 bits of padding => 15.875 bytes

	//__attribute__((aligned(16)))
	__sha256_block_t block[3];
	uint8_t *ptr = (uint8_t*)block;
	// 168 bytes of data
	memcpy((char*)ptr, (const char*)data, 168);
	ptr += 168;

	*ptr++ = 0x80; // End of string marker (and 7 bits padding)
	// Pad to (k+l+1 = 448 mod 512)
	// l = 168*8 = 1344bits
	// Blocks: 512bit | 512bit | 512bit
	// (512*3-65-l) = 1536-65-l = 1471 - l = 1471-1344 = 127bit = 15.875 bytes
	//memset(ptr, 0, 15);
	//ptr += 15;
	memset((char*)ptr, (const char)0, 21);
	ptr += 21;
	// 8 bytes is length (in bits)
	// 1344bit = 0x540
	/**ptr++ = 0x0;
	*ptr++ = 0;
	*ptr++ = 0;
	*ptr++ = 0;
	*ptr++ = 0;
	*ptr++ = 0;*/
	*ptr++ = 0x5;
	*ptr++ = 0x40;

	__sha256_hash_t ohash;
	//memcpy((char*)ohash, (const char*)__sha256_init, 32);
	for (int i = 0; i < 8; i++) {
		ohash[i] = __sha256_init[i];
	}
	device_sha256_osol(block[0], ohash);
	device_sha256_osol(block[1], ohash);
	device_sha256_osol(block[2], ohash);

	uint8_t *h = (uint8_t*)ohash;
	__global uint8_t *outp = outhash;
	for (int i = 0; i < 32/4; i++) {
		// Fix endianness at the same time
		*outp++ = h[3];
		*outp++ = h[2];
		*outp++ = h[1];
		*outp++ = h[0];
		h += 4;
	}
}

__device__ void device_sha256_generic(uint8_t *data, __global uint8_t *outhash, uint32_t len) {
#ifdef DEBUG
	if (len > 184) {
		printf("Longer than 3 blocks (184bytes), sha256_generic not made for this..\n");
		len = 184;
	}
#endif
	uint8_t num_blocks = len/64 + 1;
	uint32_t tot_len = num_blocks*512 - 65; // 64bit header
	uint32_t num_padding = (tot_len - len*8)/8;

	//__attribute__((aligned(16)))
	__sha256_block_t block[3];
	uint8_t *ptr = (uint8_t*)block;
	memcpy((char*)ptr, (const char*)data, len);
	ptr += len;

	*ptr++ = 0x80; // End of string marker (and 7 bits padding)
	// Pad to (k+l+1 = 448 mod 512)
	// l = 168*8 = 1344bits
	// Blocks: 512bit | 512bit | 512bit
	// (512*3-65-l) = 1536-65-l = 1471 - l = 1471-1344 = 127bit = 15.875 bytes
	memset(ptr, 0, num_padding);
	ptr += num_padding;
	// 8 bytes is length (in bits)
	*ptr++ = 0;
	*ptr++ = 0;
	*ptr++ = 0;
	*ptr++ = 0;
	*ptr++ = 0;
	*ptr++ = 0;
	*ptr++ = ((len * 8) & 0xff00) >> 8;
	*ptr++ = (len * 8) & 0xff;

	__sha256_hash_t ohash;
	//memcpy32((uint32_t*)ohash, (const uint32_t*)__sha256_init, 8);
	for (int i = 0; i < 8; i++) {
		ohash[i] = __sha256_init[i];
	}
	for (int i = 0; i < num_blocks; i++) {
		device_sha256_osol(block[i], ohash);
	}

	uint8_t *h = (uint8_t*)ohash;
	__global uint8_t *outp = outhash;
	for (int i = 0; i < 32/4; i++) {
		// Fix endianness at the same time
		*outp++ = h[3];
		*outp++ = h[2];
		*outp++ = h[1];
		*outp++ = h[0];
		h += 4;
	}
}


/**************** SHA256 from github sha256-sse ***************/

#define	SHA256_CONST(x)		(SHA256_CONST_ ## x)

/* constants, as provided in FIPS 180-2 */

#define	SHA256_CONST_0		0x428a2f98U
#define	SHA256_CONST_1		0x71374491U
#define	SHA256_CONST_2		0xb5c0fbcfU
#define	SHA256_CONST_3		0xe9b5dba5U
#define	SHA256_CONST_4		0x3956c25bU
#define	SHA256_CONST_5		0x59f111f1U
#define	SHA256_CONST_6		0x923f82a4U
#define	SHA256_CONST_7		0xab1c5ed5U

#define	SHA256_CONST_8		0xd807aa98U
#define	SHA256_CONST_9		0x12835b01U
#define	SHA256_CONST_10		0x243185beU
#define	SHA256_CONST_11		0x550c7dc3U
#define	SHA256_CONST_12		0x72be5d74U
#define	SHA256_CONST_13		0x80deb1feU
#define	SHA256_CONST_14		0x9bdc06a7U
#define	SHA256_CONST_15		0xc19bf174U

#define	SHA256_CONST_16		0xe49b69c1U
#define	SHA256_CONST_17		0xefbe4786U
#define	SHA256_CONST_18		0x0fc19dc6U
#define	SHA256_CONST_19		0x240ca1ccU
#define	SHA256_CONST_20		0x2de92c6fU
#define	SHA256_CONST_21		0x4a7484aaU
#define	SHA256_CONST_22		0x5cb0a9dcU
#define	SHA256_CONST_23		0x76f988daU

#define	SHA256_CONST_24		0x983e5152U
#define	SHA256_CONST_25		0xa831c66dU
#define	SHA256_CONST_26		0xb00327c8U
#define	SHA256_CONST_27		0xbf597fc7U
#define	SHA256_CONST_28		0xc6e00bf3U
#define	SHA256_CONST_29		0xd5a79147U
#define	SHA256_CONST_30		0x06ca6351U
#define	SHA256_CONST_31		0x14292967U

#define	SHA256_CONST_32		0x27b70a85U
#define	SHA256_CONST_33		0x2e1b2138U
#define	SHA256_CONST_34		0x4d2c6dfcU
#define	SHA256_CONST_35		0x53380d13U
#define	SHA256_CONST_36		0x650a7354U
#define	SHA256_CONST_37		0x766a0abbU
#define	SHA256_CONST_38		0x81c2c92eU
#define	SHA256_CONST_39		0x92722c85U

#define	SHA256_CONST_40		0xa2bfe8a1U
#define	SHA256_CONST_41		0xa81a664bU
#define	SHA256_CONST_42		0xc24b8b70U
#define	SHA256_CONST_43		0xc76c51a3U
#define	SHA256_CONST_44		0xd192e819U
#define	SHA256_CONST_45		0xd6990624U
#define	SHA256_CONST_46		0xf40e3585U
#define	SHA256_CONST_47		0x106aa070U

#define	SHA256_CONST_48		0x19a4c116U
#define	SHA256_CONST_49		0x1e376c08U
#define	SHA256_CONST_50		0x2748774cU
#define	SHA256_CONST_51		0x34b0bcb5U
#define	SHA256_CONST_52		0x391c0cb3U
#define	SHA256_CONST_53		0x4ed8aa4aU
#define	SHA256_CONST_54		0x5b9cca4fU
#define	SHA256_CONST_55		0x682e6ff3U

#define	SHA256_CONST_56		0x748f82eeU
#define	SHA256_CONST_57		0x78a5636fU
#define	SHA256_CONST_58		0x84c87814U
#define	SHA256_CONST_59		0x8cc70208U
#define	SHA256_CONST_60		0x90befffaU
#define	SHA256_CONST_61		0xa4506cebU
#define	SHA256_CONST_62		0xbef9a3f7U
#define	SHA256_CONST_63		0xc67178f2U

/* Ch and Maj are the basic SHA2 functions. */
#define	Ch(b, c, d)	(((b) & (c)) ^ ((~b) & (d)))
#define	Maj(b, c, d)	(((b) & (c)) ^ ((b) & (d)) ^ ((c) & (d)))

/* Rotates x right n bits. */
#define	ROTR(x, n)	\
(((x) >> (n)) | ((x) << ((sizeof (x) * 8)-(n))))

/* Shift x right n bits */
#define	SHR(x, n)	((x) >> (n))

/* SHA256 Functions */
#define	BIGSIGMA0_256(x)	(ROTR((x), 2) ^ ROTR((x), 13) ^ ROTR((x), 22))
#define	BIGSIGMA1_256(x)	(ROTR((x), 6) ^ ROTR((x), 11) ^ ROTR((x), 25))
#define	SIGMA0_256(x)		(ROTR((x), 7) ^ ROTR((x), 18) ^ SHR((x), 3))
#define	SIGMA1_256(x)		(ROTR((x), 17) ^ ROTR((x), 19) ^ SHR((x), 10))

#define	SHA256ROUND(a, b, c, d, e, f, g, h, i, w)			\
T1 = h + BIGSIGMA1_256(e) + Ch(e, f, g) + SHA256_CONST(i) + w;	\
d += T1;							\
T2 = BIGSIGMA0_256(a) + Maj(a, b, c);				\
h = T1 + T2


/*
 * sparc optimization:
 *
 * on the sparc, we can load big endian 32-bit data easily.  note that
 * special care must be taken to ensure the address is 32-bit aligned.
 * in the interest of speed, we don't check to make sure, since
 * careful programming can guarantee this for us.
 */

#if	defined(_BIG_ENDIAN)
#define	LOAD_BIG_32(addr)	(*(uint32_t *)(addr))
#define	LOAD_BIG_64(addr)	(*(uint64_t *)(addr))

#elif	defined(HAVE_HTONL)
#define	LOAD_BIG_32(addr) htonl(*((uint32_t *)(addr)))
#define	LOAD_BIG_64(addr) htonll(*((uint64_t *)(addr)))

#else
/* little endian -- will work on big endian, but slowly */
#define	LOAD_BIG_32(addr)	\
(((addr)[0] << 24) | ((addr)[1] << 16) | ((addr)[2] << 8) | (addr)[3])
#define	LOAD_BIG_64(addr)	\
(((uint64_t)(addr)[0] << 56) | ((uint64_t)(addr)[1] << 48) |	\
((uint64_t)(addr)[2] << 40) | ((uint64_t)(addr)[3] << 32) |	\
((uint64_t)(addr)[4] << 24) | ((uint64_t)(addr)[5] << 16) |	\
((uint64_t)(addr)[6] << 8) | (uint64_t)(addr)[7])
#endif	/* _BIG_ENDIAN */

#if 0
#define dumpstate() printf("%s: %08x %08x %08x %08x %08x %08x %08x %08x %08x\n", __func__, w0, a, b, c, d, e, f, g, h);
#else
#define dumpstate()
#endif
void host_sha256_osol(const __sha256_block_t blk, __sha256_hash_t ctx) {
	uint32_t a = ctx[0];
	uint32_t b = ctx[1];
	uint32_t c = ctx[2];
	uint32_t d = ctx[3];
	uint32_t e = ctx[4];
	uint32_t f = ctx[5];
	uint32_t g = ctx[6];
	uint32_t h = ctx[7];

	uint32_t w0, w1, w2, w3, w4, w5, w6, w7;
	uint32_t w8, w9, w10, w11, w12, w13, w14, w15;
	uint32_t T1, T2;

	/* LINTED E_BAD_PTR_CAST_ALIGN */
	w0 =  LOAD_BIG_32(blk + 4 * 0);
    dumpstate();
	SHA256ROUND(a, b, c, d, e, f, g, h, 0, w0);
    dumpstate();

	/* LINTED E_BAD_PTR_CAST_ALIGN */
	w1 =  LOAD_BIG_32(blk + 4 * 1);
	SHA256ROUND(h, a, b, c, d, e, f, g, 1, w1);
	/* LINTED E_BAD_PTR_CAST_ALIGN */
	w2 =  LOAD_BIG_32(blk + 4 * 2);
	SHA256ROUND(g, h, a, b, c, d, e, f, 2, w2);
	/* LINTED E_BAD_PTR_CAST_ALIGN */
	w3 =  LOAD_BIG_32(blk + 4 * 3);
	SHA256ROUND(f, g, h, a, b, c, d, e, 3, w3);
	/* LINTED E_BAD_PTR_CAST_ALIGN */
	w4 =  LOAD_BIG_32(blk + 4 * 4);
	SHA256ROUND(e, f, g, h, a, b, c, d, 4, w4);
	/* LINTED E_BAD_PTR_CAST_ALIGN */
	w5 =  LOAD_BIG_32(blk + 4 * 5);
	SHA256ROUND(d, e, f, g, h, a, b, c, 5, w5);
	/* LINTED E_BAD_PTR_CAST_ALIGN */
	w6 =  LOAD_BIG_32(blk + 4 * 6);
	SHA256ROUND(c, d, e, f, g, h, a, b, 6, w6);
	/* LINTED E_BAD_PTR_CAST_ALIGN */
	w7 =  LOAD_BIG_32(blk + 4 * 7);
	SHA256ROUND(b, c, d, e, f, g, h, a, 7, w7);
	/* LINTED E_BAD_PTR_CAST_ALIGN */
	w8 =  LOAD_BIG_32(blk + 4 * 8);
	SHA256ROUND(a, b, c, d, e, f, g, h, 8, w8);
	/* LINTED E_BAD_PTR_CAST_ALIGN */
	w9 =  LOAD_BIG_32(blk + 4 * 9);
	SHA256ROUND(h, a, b, c, d, e, f, g, 9, w9);
	/* LINTED E_BAD_PTR_CAST_ALIGN */
	w10 =  LOAD_BIG_32(blk + 4 * 10);
	SHA256ROUND(g, h, a, b, c, d, e, f, 10, w10);
	/* LINTED E_BAD_PTR_CAST_ALIGN */
	w11 =  LOAD_BIG_32(blk + 4 * 11);
	SHA256ROUND(f, g, h, a, b, c, d, e, 11, w11);
	/* LINTED E_BAD_PTR_CAST_ALIGN */
	w12 =  LOAD_BIG_32(blk + 4 * 12);
	SHA256ROUND(e, f, g, h, a, b, c, d, 12, w12);
	/* LINTED E_BAD_PTR_CAST_ALIGN */
	w13 =  LOAD_BIG_32(blk + 4 * 13);
	SHA256ROUND(d, e, f, g, h, a, b, c, 13, w13);
	/* LINTED E_BAD_PTR_CAST_ALIGN */
	w14 =  LOAD_BIG_32(blk + 4 * 14);
	SHA256ROUND(c, d, e, f, g, h, a, b, 14, w14);
	/* LINTED E_BAD_PTR_CAST_ALIGN */
	w15 =  LOAD_BIG_32(blk + 4 * 15);
	SHA256ROUND(b, c, d, e, f, g, h, a, 15, w15);

	w0 = SIGMA1_256(w14) + w9 + SIGMA0_256(w1) + w0;
	SHA256ROUND(a, b, c, d, e, f, g, h, 16, w0);
	w1 = SIGMA1_256(w15) + w10 + SIGMA0_256(w2) + w1;
	SHA256ROUND(h, a, b, c, d, e, f, g, 17, w1);
	w2 = SIGMA1_256(w0) + w11 + SIGMA0_256(w3) + w2;
	SHA256ROUND(g, h, a, b, c, d, e, f, 18, w2);
	w3 = SIGMA1_256(w1) + w12 + SIGMA0_256(w4) + w3;
	SHA256ROUND(f, g, h, a, b, c, d, e, 19, w3);
	w4 = SIGMA1_256(w2) + w13 + SIGMA0_256(w5) + w4;
	SHA256ROUND(e, f, g, h, a, b, c, d, 20, w4);
	w5 = SIGMA1_256(w3) + w14 + SIGMA0_256(w6) + w5;
	SHA256ROUND(d, e, f, g, h, a, b, c, 21, w5);
	w6 = SIGMA1_256(w4) + w15 + SIGMA0_256(w7) + w6;
	SHA256ROUND(c, d, e, f, g, h, a, b, 22, w6);
	w7 = SIGMA1_256(w5) + w0 + SIGMA0_256(w8) + w7;
	SHA256ROUND(b, c, d, e, f, g, h, a, 23, w7);
	w8 = SIGMA1_256(w6) + w1 + SIGMA0_256(w9) + w8;
	SHA256ROUND(a, b, c, d, e, f, g, h, 24, w8);
	w9 = SIGMA1_256(w7) + w2 + SIGMA0_256(w10) + w9;
	SHA256ROUND(h, a, b, c, d, e, f, g, 25, w9);
	w10 = SIGMA1_256(w8) + w3 + SIGMA0_256(w11) + w10;
	SHA256ROUND(g, h, a, b, c, d, e, f, 26, w10);
	w11 = SIGMA1_256(w9) + w4 + SIGMA0_256(w12) + w11;
	SHA256ROUND(f, g, h, a, b, c, d, e, 27, w11);
	w12 = SIGMA1_256(w10) + w5 + SIGMA0_256(w13) + w12;
	SHA256ROUND(e, f, g, h, a, b, c, d, 28, w12);
	w13 = SIGMA1_256(w11) + w6 + SIGMA0_256(w14) + w13;
	SHA256ROUND(d, e, f, g, h, a, b, c, 29, w13);
	w14 = SIGMA1_256(w12) + w7 + SIGMA0_256(w15) + w14;
	SHA256ROUND(c, d, e, f, g, h, a, b, 30, w14);
	w15 = SIGMA1_256(w13) + w8 + SIGMA0_256(w0) + w15;
	SHA256ROUND(b, c, d, e, f, g, h, a, 31, w15);

	w0 = SIGMA1_256(w14) + w9 + SIGMA0_256(w1) + w0;
	SHA256ROUND(a, b, c, d, e, f, g, h, 32, w0);
	w1 = SIGMA1_256(w15) + w10 + SIGMA0_256(w2) + w1;
	SHA256ROUND(h, a, b, c, d, e, f, g, 33, w1);
	w2 = SIGMA1_256(w0) + w11 + SIGMA0_256(w3) + w2;
	SHA256ROUND(g, h, a, b, c, d, e, f, 34, w2);
	w3 = SIGMA1_256(w1) + w12 + SIGMA0_256(w4) + w3;
	SHA256ROUND(f, g, h, a, b, c, d, e, 35, w3);
	w4 = SIGMA1_256(w2) + w13 + SIGMA0_256(w5) + w4;
	SHA256ROUND(e, f, g, h, a, b, c, d, 36, w4);
	w5 = SIGMA1_256(w3) + w14 + SIGMA0_256(w6) + w5;
	SHA256ROUND(d, e, f, g, h, a, b, c, 37, w5);
	w6 = SIGMA1_256(w4) + w15 + SIGMA0_256(w7) + w6;
	SHA256ROUND(c, d, e, f, g, h, a, b, 38, w6);
	w7 = SIGMA1_256(w5) + w0 + SIGMA0_256(w8) + w7;
	SHA256ROUND(b, c, d, e, f, g, h, a, 39, w7);
	w8 = SIGMA1_256(w6) + w1 + SIGMA0_256(w9) + w8;
	SHA256ROUND(a, b, c, d, e, f, g, h, 40, w8);
	w9 = SIGMA1_256(w7) + w2 + SIGMA0_256(w10) + w9;
	SHA256ROUND(h, a, b, c, d, e, f, g, 41, w9);
	w10 = SIGMA1_256(w8) + w3 + SIGMA0_256(w11) + w10;
	SHA256ROUND(g, h, a, b, c, d, e, f, 42, w10);
	w11 = SIGMA1_256(w9) + w4 + SIGMA0_256(w12) + w11;
	SHA256ROUND(f, g, h, a, b, c, d, e, 43, w11);
	w12 = SIGMA1_256(w10) + w5 + SIGMA0_256(w13) + w12;
	SHA256ROUND(e, f, g, h, a, b, c, d, 44, w12);
	w13 = SIGMA1_256(w11) + w6 + SIGMA0_256(w14) + w13;
	SHA256ROUND(d, e, f, g, h, a, b, c, 45, w13);
	w14 = SIGMA1_256(w12) + w7 + SIGMA0_256(w15) + w14;
	SHA256ROUND(c, d, e, f, g, h, a, b, 46, w14);
	w15 = SIGMA1_256(w13) + w8 + SIGMA0_256(w0) + w15;
	SHA256ROUND(b, c, d, e, f, g, h, a, 47, w15);

	w0 = SIGMA1_256(w14) + w9 + SIGMA0_256(w1) + w0;
	SHA256ROUND(a, b, c, d, e, f, g, h, 48, w0);
	w1 = SIGMA1_256(w15) + w10 + SIGMA0_256(w2) + w1;
	SHA256ROUND(h, a, b, c, d, e, f, g, 49, w1);
	w2 = SIGMA1_256(w0) + w11 + SIGMA0_256(w3) + w2;
	SHA256ROUND(g, h, a, b, c, d, e, f, 50, w2);
	w3 = SIGMA1_256(w1) + w12 + SIGMA0_256(w4) + w3;
	SHA256ROUND(f, g, h, a, b, c, d, e, 51, w3);
	w4 = SIGMA1_256(w2) + w13 + SIGMA0_256(w5) + w4;
	SHA256ROUND(e, f, g, h, a, b, c, d, 52, w4);
	w5 = SIGMA1_256(w3) + w14 + SIGMA0_256(w6) + w5;
	SHA256ROUND(d, e, f, g, h, a, b, c, 53, w5);
	w6 = SIGMA1_256(w4) + w15 + SIGMA0_256(w7) + w6;
	SHA256ROUND(c, d, e, f, g, h, a, b, 54, w6);
	w7 = SIGMA1_256(w5) + w0 + SIGMA0_256(w8) + w7;
	SHA256ROUND(b, c, d, e, f, g, h, a, 55, w7);
	w8 = SIGMA1_256(w6) + w1 + SIGMA0_256(w9) + w8;
	SHA256ROUND(a, b, c, d, e, f, g, h, 56, w8);
	w9 = SIGMA1_256(w7) + w2 + SIGMA0_256(w10) + w9;
	SHA256ROUND(h, a, b, c, d, e, f, g, 57, w9);
	w10 = SIGMA1_256(w8) + w3 + SIGMA0_256(w11) + w10;
	SHA256ROUND(g, h, a, b, c, d, e, f, 58, w10);
	w11 = SIGMA1_256(w9) + w4 + SIGMA0_256(w12) + w11;
	SHA256ROUND(f, g, h, a, b, c, d, e, 59, w11);
	w12 = SIGMA1_256(w10) + w5 + SIGMA0_256(w13) + w12;
	SHA256ROUND(e, f, g, h, a, b, c, d, 60, w12);
	w13 = SIGMA1_256(w11) + w6 + SIGMA0_256(w14) + w13;
	SHA256ROUND(d, e, f, g, h, a, b, c, 61, w13);
	w14 = SIGMA1_256(w12) + w7 + SIGMA0_256(w15) + w14;
	SHA256ROUND(c, d, e, f, g, h, a, b, 62, w14);
	w15 = SIGMA1_256(w13) + w8 + SIGMA0_256(w0) + w15;
	SHA256ROUND(b, c, d, e, f, g, h, a, 63, w15);

    //printf("%s last d: %08x\n", __func__, d);

    //printf("%s a: %08x %08x\n", __func__, a, ctx[0]);
	ctx[0] += a;
    //printf("%s a: %08x\n", __func__, ctx[0]);
	ctx[1] += b;
	ctx[2] += c;
	ctx[3] += d;
	ctx[4] += e;
	ctx[5] += f;
	ctx[6] += g;
	ctx[7] += h;

}

__device__ void device_sha256_osol(const __sha256_block_t blk, __sha256_hash_t ctx) {
	uint32_t a = ctx[0];
	uint32_t b = ctx[1];
	uint32_t c = ctx[2];
	uint32_t d = ctx[3];
	uint32_t e = ctx[4];
	uint32_t f = ctx[5];
	uint32_t g = ctx[6];
	uint32_t h = ctx[7];

	uint32_t w0, w1, w2, w3, w4, w5, w6, w7;
	uint32_t w8, w9, w10, w11, w12, w13, w14, w15;
	uint32_t T1, T2;

	/* LINTED E_BAD_PTR_CAST_ALIGN */
	w0 =  LOAD_BIG_32(blk + 4 * 0);
    dumpstate();
	SHA256ROUND(a, b, c, d, e, f, g, h, 0, w0);
    dumpstate();

	/* LINTED E_BAD_PTR_CAST_ALIGN */
	w1 =  LOAD_BIG_32(blk + 4 * 1);
	SHA256ROUND(h, a, b, c, d, e, f, g, 1, w1);
	/* LINTED E_BAD_PTR_CAST_ALIGN */
	w2 =  LOAD_BIG_32(blk + 4 * 2);
	SHA256ROUND(g, h, a, b, c, d, e, f, 2, w2);
	/* LINTED E_BAD_PTR_CAST_ALIGN */
	w3 =  LOAD_BIG_32(blk + 4 * 3);
	SHA256ROUND(f, g, h, a, b, c, d, e, 3, w3);
	/* LINTED E_BAD_PTR_CAST_ALIGN */
	w4 =  LOAD_BIG_32(blk + 4 * 4);
	SHA256ROUND(e, f, g, h, a, b, c, d, 4, w4);
	/* LINTED E_BAD_PTR_CAST_ALIGN */
	w5 =  LOAD_BIG_32(blk + 4 * 5);
	SHA256ROUND(d, e, f, g, h, a, b, c, 5, w5);
	/* LINTED E_BAD_PTR_CAST_ALIGN */
	w6 =  LOAD_BIG_32(blk + 4 * 6);
	SHA256ROUND(c, d, e, f, g, h, a, b, 6, w6);
	/* LINTED E_BAD_PTR_CAST_ALIGN */
	w7 =  LOAD_BIG_32(blk + 4 * 7);
	SHA256ROUND(b, c, d, e, f, g, h, a, 7, w7);
	/* LINTED E_BAD_PTR_CAST_ALIGN */
	w8 =  LOAD_BIG_32(blk + 4 * 8);
	SHA256ROUND(a, b, c, d, e, f, g, h, 8, w8);
	/* LINTED E_BAD_PTR_CAST_ALIGN */
	w9 =  LOAD_BIG_32(blk + 4 * 9);
	SHA256ROUND(h, a, b, c, d, e, f, g, 9, w9);
	/* LINTED E_BAD_PTR_CAST_ALIGN */
	w10 =  LOAD_BIG_32(blk + 4 * 10);
	SHA256ROUND(g, h, a, b, c, d, e, f, 10, w10);
	/* LINTED E_BAD_PTR_CAST_ALIGN */
	w11 =  LOAD_BIG_32(blk + 4 * 11);
	SHA256ROUND(f, g, h, a, b, c, d, e, 11, w11);
	/* LINTED E_BAD_PTR_CAST_ALIGN */
	w12 =  LOAD_BIG_32(blk + 4 * 12);
	SHA256ROUND(e, f, g, h, a, b, c, d, 12, w12);
	/* LINTED E_BAD_PTR_CAST_ALIGN */
	w13 =  LOAD_BIG_32(blk + 4 * 13);
	SHA256ROUND(d, e, f, g, h, a, b, c, 13, w13);
	/* LINTED E_BAD_PTR_CAST_ALIGN */
	w14 =  LOAD_BIG_32(blk + 4 * 14);
	SHA256ROUND(c, d, e, f, g, h, a, b, 14, w14);
	/* LINTED E_BAD_PTR_CAST_ALIGN */
	w15 =  LOAD_BIG_32(blk + 4 * 15);
	SHA256ROUND(b, c, d, e, f, g, h, a, 15, w15);

	w0 = SIGMA1_256(w14) + w9 + SIGMA0_256(w1) + w0;
	SHA256ROUND(a, b, c, d, e, f, g, h, 16, w0);
	w1 = SIGMA1_256(w15) + w10 + SIGMA0_256(w2) + w1;
	SHA256ROUND(h, a, b, c, d, e, f, g, 17, w1);
	w2 = SIGMA1_256(w0) + w11 + SIGMA0_256(w3) + w2;
	SHA256ROUND(g, h, a, b, c, d, e, f, 18, w2);
	w3 = SIGMA1_256(w1) + w12 + SIGMA0_256(w4) + w3;
	SHA256ROUND(f, g, h, a, b, c, d, e, 19, w3);
	w4 = SIGMA1_256(w2) + w13 + SIGMA0_256(w5) + w4;
	SHA256ROUND(e, f, g, h, a, b, c, d, 20, w4);
	w5 = SIGMA1_256(w3) + w14 + SIGMA0_256(w6) + w5;
	SHA256ROUND(d, e, f, g, h, a, b, c, 21, w5);
	w6 = SIGMA1_256(w4) + w15 + SIGMA0_256(w7) + w6;
	SHA256ROUND(c, d, e, f, g, h, a, b, 22, w6);
	w7 = SIGMA1_256(w5) + w0 + SIGMA0_256(w8) + w7;
	SHA256ROUND(b, c, d, e, f, g, h, a, 23, w7);
	w8 = SIGMA1_256(w6) + w1 + SIGMA0_256(w9) + w8;
	SHA256ROUND(a, b, c, d, e, f, g, h, 24, w8);
	w9 = SIGMA1_256(w7) + w2 + SIGMA0_256(w10) + w9;
	SHA256ROUND(h, a, b, c, d, e, f, g, 25, w9);
	w10 = SIGMA1_256(w8) + w3 + SIGMA0_256(w11) + w10;
	SHA256ROUND(g, h, a, b, c, d, e, f, 26, w10);
	w11 = SIGMA1_256(w9) + w4 + SIGMA0_256(w12) + w11;
	SHA256ROUND(f, g, h, a, b, c, d, e, 27, w11);
	w12 = SIGMA1_256(w10) + w5 + SIGMA0_256(w13) + w12;
	SHA256ROUND(e, f, g, h, a, b, c, d, 28, w12);
	w13 = SIGMA1_256(w11) + w6 + SIGMA0_256(w14) + w13;
	SHA256ROUND(d, e, f, g, h, a, b, c, 29, w13);
	w14 = SIGMA1_256(w12) + w7 + SIGMA0_256(w15) + w14;
	SHA256ROUND(c, d, e, f, g, h, a, b, 30, w14);
	w15 = SIGMA1_256(w13) + w8 + SIGMA0_256(w0) + w15;
	SHA256ROUND(b, c, d, e, f, g, h, a, 31, w15);

	w0 = SIGMA1_256(w14) + w9 + SIGMA0_256(w1) + w0;
	SHA256ROUND(a, b, c, d, e, f, g, h, 32, w0);
	w1 = SIGMA1_256(w15) + w10 + SIGMA0_256(w2) + w1;
	SHA256ROUND(h, a, b, c, d, e, f, g, 33, w1);
	w2 = SIGMA1_256(w0) + w11 + SIGMA0_256(w3) + w2;
	SHA256ROUND(g, h, a, b, c, d, e, f, 34, w2);
	w3 = SIGMA1_256(w1) + w12 + SIGMA0_256(w4) + w3;
	SHA256ROUND(f, g, h, a, b, c, d, e, 35, w3);
	w4 = SIGMA1_256(w2) + w13 + SIGMA0_256(w5) + w4;
	SHA256ROUND(e, f, g, h, a, b, c, d, 36, w4);
	w5 = SIGMA1_256(w3) + w14 + SIGMA0_256(w6) + w5;
	SHA256ROUND(d, e, f, g, h, a, b, c, 37, w5);
	w6 = SIGMA1_256(w4) + w15 + SIGMA0_256(w7) + w6;
	SHA256ROUND(c, d, e, f, g, h, a, b, 38, w6);
	w7 = SIGMA1_256(w5) + w0 + SIGMA0_256(w8) + w7;
	SHA256ROUND(b, c, d, e, f, g, h, a, 39, w7);
	w8 = SIGMA1_256(w6) + w1 + SIGMA0_256(w9) + w8;
	SHA256ROUND(a, b, c, d, e, f, g, h, 40, w8);
	w9 = SIGMA1_256(w7) + w2 + SIGMA0_256(w10) + w9;
	SHA256ROUND(h, a, b, c, d, e, f, g, 41, w9);
	w10 = SIGMA1_256(w8) + w3 + SIGMA0_256(w11) + w10;
	SHA256ROUND(g, h, a, b, c, d, e, f, 42, w10);
	w11 = SIGMA1_256(w9) + w4 + SIGMA0_256(w12) + w11;
	SHA256ROUND(f, g, h, a, b, c, d, e, 43, w11);
	w12 = SIGMA1_256(w10) + w5 + SIGMA0_256(w13) + w12;
	SHA256ROUND(e, f, g, h, a, b, c, d, 44, w12);
	w13 = SIGMA1_256(w11) + w6 + SIGMA0_256(w14) + w13;
	SHA256ROUND(d, e, f, g, h, a, b, c, 45, w13);
	w14 = SIGMA1_256(w12) + w7 + SIGMA0_256(w15) + w14;
	SHA256ROUND(c, d, e, f, g, h, a, b, 46, w14);
	w15 = SIGMA1_256(w13) + w8 + SIGMA0_256(w0) + w15;
	SHA256ROUND(b, c, d, e, f, g, h, a, 47, w15);

	w0 = SIGMA1_256(w14) + w9 + SIGMA0_256(w1) + w0;
	SHA256ROUND(a, b, c, d, e, f, g, h, 48, w0);
	w1 = SIGMA1_256(w15) + w10 + SIGMA0_256(w2) + w1;
	SHA256ROUND(h, a, b, c, d, e, f, g, 49, w1);
	w2 = SIGMA1_256(w0) + w11 + SIGMA0_256(w3) + w2;
	SHA256ROUND(g, h, a, b, c, d, e, f, 50, w2);
	w3 = SIGMA1_256(w1) + w12 + SIGMA0_256(w4) + w3;
	SHA256ROUND(f, g, h, a, b, c, d, e, 51, w3);
	w4 = SIGMA1_256(w2) + w13 + SIGMA0_256(w5) + w4;
	SHA256ROUND(e, f, g, h, a, b, c, d, 52, w4);
	w5 = SIGMA1_256(w3) + w14 + SIGMA0_256(w6) + w5;
	SHA256ROUND(d, e, f, g, h, a, b, c, 53, w5);
	w6 = SIGMA1_256(w4) + w15 + SIGMA0_256(w7) + w6;
	SHA256ROUND(c, d, e, f, g, h, a, b, 54, w6);
	w7 = SIGMA1_256(w5) + w0 + SIGMA0_256(w8) + w7;
	SHA256ROUND(b, c, d, e, f, g, h, a, 55, w7);
	w8 = SIGMA1_256(w6) + w1 + SIGMA0_256(w9) + w8;
	SHA256ROUND(a, b, c, d, e, f, g, h, 56, w8);
	w9 = SIGMA1_256(w7) + w2 + SIGMA0_256(w10) + w9;
	SHA256ROUND(h, a, b, c, d, e, f, g, 57, w9);
	w10 = SIGMA1_256(w8) + w3 + SIGMA0_256(w11) + w10;
	SHA256ROUND(g, h, a, b, c, d, e, f, 58, w10);
	w11 = SIGMA1_256(w9) + w4 + SIGMA0_256(w12) + w11;
	SHA256ROUND(f, g, h, a, b, c, d, e, 59, w11);
	w12 = SIGMA1_256(w10) + w5 + SIGMA0_256(w13) + w12;
	SHA256ROUND(e, f, g, h, a, b, c, d, 60, w12);
	w13 = SIGMA1_256(w11) + w6 + SIGMA0_256(w14) + w13;
	SHA256ROUND(d, e, f, g, h, a, b, c, 61, w13);
	w14 = SIGMA1_256(w12) + w7 + SIGMA0_256(w15) + w14;
	SHA256ROUND(c, d, e, f, g, h, a, b, 62, w14);
	w15 = SIGMA1_256(w13) + w8 + SIGMA0_256(w0) + w15;
	SHA256ROUND(b, c, d, e, f, g, h, a, 63, w15);

    //printf("%s last d: %08x\n", __func__, d);

    //printf("%s a: %08x %08x\n", __func__, a, ctx[0]);
	ctx[0] += a;
    //printf("%s a: %08x\n", __func__, ctx[0]);
	ctx[1] += b;
	ctx[2] += c;
	ctx[3] += d;
	ctx[4] += e;
	ctx[5] += f;
	ctx[6] += g;
	ctx[7] += h;

}
