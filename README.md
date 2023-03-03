# `blc`: A compiler for a [B](https://en.wikipedia.org/wiki/B_(programming_language))-like language

This originally was course work for the [Code Generation for Data Processing](https://db.in.tum.de/teaching/ws2223/codegen/?lang=en) course at TUM.
After finishing the course, I found myself wanting to improve and expand the compiler further.

# Language specification
## Syntax

    program: function*
    function: ident "(" paramlist? ")" block
    paramlist: ident ("," ident)*
    block: "{" statement* "}"
    statement: ("auto" | "register") ident "=" expr ";"
             | "return" expr? ";"
             | block
             | "while" "(" expr ")" statement
             | "if" "(" expr ")" statement ("else" statement)?
             | expr ";"
    expr: "(" expr ")"
        | ident "(" arglist? ")"
        | ident
        | number
        | "-" expr
        | "!" expr
        | "~" expr
        | "&" expr
        | expr "[" expr sizespec? "]"
        | expr "+" expr
        | expr "-" expr
        | expr "*" expr
        | expr "/" expr
        | expr "%" expr
        | expr "<<" expr
        | expr ">>" expr
        | expr "<" expr
        | expr ">" expr
        | expr "<=" expr
        | expr ">=" expr
        | expr "==" expr
        | expr "!=" expr
        | expr "&" expr
        | expr "|" expr
        | expr "^" expr
        | expr "&&" expr
        | expr "||" expr
        | expr "=" expr
    arglist: expr ("," expr)*
    sizespec: "@" number
    ident: r"[a-zA-Z_][a-zA-Z0-9_]*"
    number: r"[0-9]+"

Additional specifications:

- Comments start with `//` and are ignored until the end of line.
- Whitespace is ignored, except to separate tokens (like C).
- Multi-char operators like `||` are never treated as `|` `|`, unless where is whitespace in between (like C).
- Keywords are `auto` `register` `if` `else` `while` `return`; they are never identifiers.
- The callee of function calls must be an unparenthesized identifier and never refers to a variable, even if a variable with the same name exists (*un*like C).
- For identifiers used to refer to variables, the variable must be declared in the current or a parent block.
- Operator precedence and associativity:

        14, left-assoc:  []
        13, right-assoc: all unary operators (- ! ~ &)
        12, left-assoc:  * / %
        11, left-assoc:  + -
        10, left-assoc:  << >>
        9,  left-assoc:  < > <= >=
        8,  left-assoc:  == !=
        7,  left-assoc:  &
        6,  left-assoc:  ^
        5,  left-assoc:  |
        4,  left-assoc:  &&
        3,  left-assoc:  ||
        1,  right-assoc: =

- The left-hand side of the assignment operator (`=`) and the operand of the address-of operator (unary `&`) must be a subscript (`a[b]`) or an identifier.
- Variables declared with `register` and parameters are not permitted as operand of the address-of operator (unary `&`).
- Variables are only accessible in their scope of declaration or enclosed blocks; parameters are accessible in the entire function.
- Valid size specifiers for subscripts are 1, 2, 4, and 8; if omitted, it defaults to 8.

## Semantics

- The signed integer datatype is 64-bit sized, other data types do not exist.
- One byte has 8 bits; the byte order is little endian.
- A subscript interprets the left operand as address and the right operand inside the brackets as offset, which is scaled by the size specifier. I.e., `a[b@X]` refers to the X bytes at address `a + X*b`. When loading from a reduced size (i.e., 1/2/4), as many bytes as specified are loaded from memory and sign-extend to a full integer; memory stores truncate the operand to the specified size. Note that therefore `a[b]` is not equivalent to `b[a]` (unlike C).
- Integer overflow is defined as in two's complement (like `-fwrapv` in GCC).
- Everything else is handled as in ANSI C.

## Examples
Example programs can be found in `bSamples`, as well as in the `tests` directory (partly contain duplicate programs)

# Building
Ensure `gcc`>=12 and LLVM 15 are installed and `llvm-config` is configured to use LLVM 15.

To build, simply use `make`.

If you haven't installed LLVM 15 system-wide, either use `LLVM_CONFIG=/path/to/llvm-config make [...]` or `export LLVM_CONFIG=/path/to/llvm-config` and then `make -e [...]`.

A debug build with assertions and debuginfo is available through `make debug`

# Tests
To run the tests using [`lit`, the LLVM Integrated Tester](https://www.llvm.org/docs/CommandGuide/lit.html) and [FileCheck](https://www.llvm.org/docs/CommandGuide/FileCheck.html), use `lit -svj1 .`. It is recommended to do this using the debug build, to utilize all assertions to catch internal errors. A `test` make target is available to do these steps automatically: `make test`.

Note: Running the tests in parallel (i.e. without `-j1`) can give unexpected results.

# Limitations

- There is only one handwritten backend, targeting AArch64 (although there is an option to simply compile using LLVM as the backend).
- The backend does not use a real proper IR (something like MIR etc.), but simply encodes instructions as calls to LLVM functions which represent instructions. This is obviously not ideal, but was done in order to save time and not have to design a whole new IR. Might be overhauled in the future.
- The backend is not at all correct yet. The frontend seems to work pretty well, but the backend certainly has bugs, at least for the moment. This is still a heavy WIP.
- Currently, the entire source is in one file. This just simplifies everything, obviating the need to repeat the declaration of every function once in a header and once in an out-of-line definition. This is a limitation, because it results in atrocious compile times (~30 seconds usually). It might be possible to separate stuff into a few files though, not one for every class, but maybe frontend.cpp/backend.cpp, to try to cut down compile times at least by a bit.

