#ifndef B_COMBINE
#define B_COMBINE

include "mlir/IR/PatternBase.td"
include "B/BOps.td"

// these are just for trying out the pattern matching stuff, I am basically sure that these roundtrips can't even occur in the generation in the first place.

def PointerIntRoundTripPattern : Pat<(IntToPtrOp (PtrToIntOp $ptr)),
                                  (replaceWithValue $ptr)>;

def IntPointerRoundTripPattern : Pat<(PtrToIntOp (IntToPtrOp $intt)),
                                  (replaceWithValue $intt)>;


#endif
