# Cuda Prep

## Simulation Parameters

Simulation Parameters will all be pointers to arrays. 
1. Pointers need to be allocated on device.
2. Memory from host copied to device pointers.
3. Device Pointers:
   1. Scalars: double scalars [NScalars]
   2. Lookup Table: unsigned long long lut[Bx By Bz]
   3. Index Array: unsigned long long index[M]
   4. Pairs: unsigned long long pairs[2 EdgeCount]
   5. Swc: double swc[4*NodeCount]
   6. Bounds: int bounds[3]
   7. Pair Bounds: int pairbounds[3 2 EdgeCount]

4. Indexing Device Pointers
    1. Scalars: Each Constant in will be set to variable in Kernel. No Indexing Afterwards
    2. Lookup Table: Indexing Based on Bound Size
    3. Index Array: Nontrivial Indexing, needs another variable to be passed storing start of each element.
    4. Pairs: 
       1. childindex = pairs[i+0]
       2. parentindex = pairs[i+1]
    5. Swc:  
       1. x = swc[i+0]
       2. y = swc[i+1]
       3. z = swc[i+2]
       4. r = swc[i+3]
    6. Bounds: Simple.
    7. Pair Bounds: 
       1. x0 = pairbounds[i + 0]
       2. y0 = pairbounds[i + 1]
       3. z0 = pairbounds[i + 2]
       4. x1 = pairbounds[i + 3]
       5. y1 = pairbounds[i + 4]
       6. z1 = pairbounds[i + 5]

***INDEXING NEEDS VERIFICATION***

