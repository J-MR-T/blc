// RUN: cc %S/euler.c -o %t-euler && %blc -l %s %t && %t | diff - <(%t-euler)
// RUN: %blc %s 2>&1 | not grep Warn

// from https://en.wikipedia.org/wiki/B_(programming_language)

main() {
    auto n = 2000;
    auto v = calloc(n, 8);
    auto i = 0;
    auto col = 0;
    auto c = 0;
    auto a = 0;

    i = col = 0;
    while(i<n){
        v[i] = 1;
        i=i+1;
    }
    while(col<2*n) {
        a = n+1;
        c = i = 0;
        while (i<n) {
            c = c + v[i] *10;
            v[i]  = c%a;
            i=i+1;
            c = c/ a;
            a=a-1;
        }

        putchar(c+48); // 48=='0'
        col = col +1;
        if(!(col%5)){
            //putchar(col%50?' ': '*n');
            if(col%50){
                putchar(20); // ' '
            }else{
                putchar(10); // '\n'
            }

        }
    }
    putchar(10);
    putchar(10);
    return 0;
}
