// RUN: %blc -l %s %t && %t | FileCheck %s
// RUN: %blc %s 2>&1 | not grep Warn
// RUN: %blc -a %s | aarch64-linux-gnu-gcc -g -x assembler -o %t - && qemu-aarch64 -L /usr/aarch64-linux-gnu %t | FileCheck %s

// CHECK-NOT: Warning
main(){
    register fmt = calloc(16,1);
    fmt[0@1] = 48; // 0
    fmt[1@1] = 120; // x
    fmt[2@1] = 37; //  %
    fmt[3@1] = 48; // 0
    fmt[4@1] = 49; // 1
    fmt[5@1] = 54; // 6
    fmt[6@1] = 108; // l
    fmt[7@1] = 108; // l
    fmt[8@1] = 120; // x
    fmt[9@1] = 10; // \n



    auto a = 72340172838076673; // (0x0101010101010101)
    register addr = &a;
    
    printf(fmt, addr[0@8]); // CHECK: 0x0101010101010101

    printf(fmt, addr[0@4]); // CHECK-NEXT: 0x0000000001010101
    printf(fmt, addr[1@4]); // CHECK-NEXT: 0x0000000001010101

    printf(fmt, addr[0@2]); // CHECK-NEXT: 0x0000000000000101
    printf(fmt, addr[1@2]); // CHECK-NEXT: 0x0000000000000101
    printf(fmt, addr[2@2]); // CHECK-NEXT: 0x0000000000000101
    printf(fmt, addr[3@2]); // CHECK-NEXT: 0x0000000000000101

    printf(fmt, addr[0@1]); // CHECK-NEXT: 0x0000000000000001
    printf(fmt, addr[1@1]); // CHECK-NEXT: 0x0000000000000001
    printf(fmt, addr[2@1]); // CHECK-NEXT: 0x0000000000000001
    printf(fmt, addr[3@1]); // CHECK-NEXT: 0x0000000000000001
    printf(fmt, addr[4@1]); // CHECK-NEXT: 0x0000000000000001
    printf(fmt, addr[5@1]); // CHECK-NEXT: 0x0000000000000001
    printf(fmt, addr[6@1]); // CHECK-NEXT: 0x0000000000000001
    printf(fmt, addr[7@1]); // CHECK-NEXT: 0x0000000000000001


    return 0;
}
