// TODO this has a weird poison in it
addrof(ptr) {
    auto var = 1;
    ptr[1] = &var;
    register ptr2 = &ptr[(-5)@4];
    ptr2 = ptr2 + 1000;
    ptr2[0] = 2;
    return var;
}

