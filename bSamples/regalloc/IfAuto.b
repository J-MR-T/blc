
fn(x, y){
    auto z = x;
    auto a = 0;
    if((a = z*x) > y){
        z = 55;
        return a;
    }else{
        return fn(a, y);
    }
}
