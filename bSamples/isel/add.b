foo(x,y){
    register a = x+(y << 2);
    register b = (x << 2) + y;
    register c = x - (y << 2);
    
    register d = x+(y << x);
    register e = (x << x) + y;
    register f = x - (y << x);

    register g = x + 5;
    register h = 5 + x;

    register i = x - 5;
    register j = 5 - x;

    return a + b + c + d + e + f + g + h + i + j;
}
