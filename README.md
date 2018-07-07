## clballoon
_opencl miner supporting balloon128/4_

### credits
cpuminer-multi, authored by tpruvot

modified for balloon 128/4, authored by barrystyle

optimized balloon, authored by Belgarion ( accepting donations at: (deft) dJP7aS2GVbmSKbHrS9aRFycdab5UNd4zxa )

Cuda conversion, authored by Belgarion

OpenCL conversion, authored by Belgarion

### installation
Very experimental, don't expect it to work properly on your system.

Tests on RX570 so far gives around 0.8 - 1.3kH/s.

Do _NOT_ modify Makefile, compilation will fail (or you will not get correct results at least).
Do _NOT_ run build.sh (since it modifies Makefile).

Compile with: make
Run with: ./cpuminer -t 1 -a balloon --cuda_threads 256 --cuda_blocks 96 -o stratum+tcp://pool:1234 -u walletaddr
