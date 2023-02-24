main(argc, argv){
    auto fmtString = calloc(1, 4); // write %s\n\0 to it
    fmtString[0@1] = 37; // %
    fmtString[1@1] = 115; // s
    fmtString[2@1] = 10; // \n
    // 0 implicit because calloc

    if(argc<=1){
        return 1;
    }
    // does printf(fmtString, argv[1])
    auto ptrPtr = argv;
    auto ptrPtrPtr = &ptrPtr;
    ptrPtr = ptrPtrPtr[0]+8;
    printf(fmtString, ptrPtr[0]);
    return 0;
}
