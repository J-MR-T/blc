#include "util.h"

namespace ArgParse{

std::map<Arg, string> parsedArgs{};

void printHelp(const char *argv0) {
    std::cerr << "A Compiler for a B like language" << std::endl;
    std::cerr << "Usage: " << std::endl;
    for (auto &arg : args) {
        std::cerr << "  ";
        if (arg.shortOpt != "")
            std::cerr << "-" << arg.shortOpt;

        if (arg.longOpt != "") {
            if (arg.shortOpt != "")
                std::cerr << ", ";

            std::cerr << "--" << arg.longOpt;
        }

        if (arg.pos != 0)
            std::cerr << " (or positional, at position " << arg.pos << ")";
        else if (arg.flag)
            std::cerr << " (flag)";

        std::cerr << "\n    "
            // string replace all \n with \n \t here
            << std::regex_replace(arg.description, std::regex("\n"), "\n    ")
            << std::endl;
    }

    std::cerr << "\nExamples: \n"
        << "  " << argv0 << " -i input.b -d -o output.dot\n"
        << "  " << argv0 << " input.b -d output.dot\n"
        << "  " << argv0 << " input.b -du\n"
        << "  " << argv0 << " -lE input.b\n"
        << "  " << argv0 << " -l main.b main\n"
        << "  " << argv0 << " -ls input.b\n"
        << "  " << argv0 << " -sr input.b\n"
        << "  " << argv0
        << " -a bSamples/asm/addressCalculations.b | aarch64-linux-gnu-gcc "
        "-g -x assembler -o test - && qemu-aarch64 -L "
        "/usr/aarch64-linux-gnu test hi\\ there\n";
}

std::map<Arg, std::string>& parse(int argc, char *argv[]) {
    std::stringstream ss;
    ss << " ";
    for (int i = 1; i < argc; ++i) {
        ss << argv[i] << " ";
    }

    string argString = ss.str();

    // handle positional args first, they have lower precedence
    // find them all, put them into a vector, then match them to the possible args
    std::vector<string> positionalArgs{};
    for (int i = 1; i < argc; ++i) {
        for (const auto &arg : args) {
            if (!arg.flag && (("-" + arg.shortOpt) == string{argv[i - 1]} ||
                        ("--" + arg.longOpt) == string{argv[i - 1]})) {
                // the current arg is the value to another argument, so we dont count it
                goto cont;
            }
        }

        if (argv[i][0] != '-') {
            // now we know its a positional arg
            positionalArgs.emplace_back(argv[i]);
        }
cont:
        continue;
    }

    for (const auto &arg : args) {
        if (arg.pos != 0) {
            // this is a positional arg
            if (positionalArgs.size() > arg.pos - 1) {
                parsedArgs[arg] = positionalArgs[arg.pos - 1];
            }
        }
    }

    bool missingRequired = false;

    // long/short/flags
    for (const auto &arg : args) {
        if (!arg.flag) {
            std::regex matchShort{" -" + arg.shortOpt + "\\s*([^\\s]+)"};
            std::regex matchLong{" --" + arg.longOpt + "(\\s*|=)([^\\s=]+)"};
            std::smatch match;
            if (arg.shortOpt != "" &&
                    std::regex_search(argString, match, matchShort)) {
                parsedArgs[arg] = match[1];
            } else if (arg.longOpt != "" &&
                    std::regex_search(argString, match, matchLong)) {
                parsedArgs[arg] = match[2];
            } else if (arg.required && !parsedArgs.contains(arg)) {
                std::cerr << "Missing required argument: -" << arg.shortOpt << "/--"
                    << arg.longOpt << std::endl;
                missingRequired = true;
            }
        } else {
            std::regex matchFlagShort{" -[a-zA-z]*" + arg.shortOpt};
            std::regex matchFlagLong{" --" + arg.longOpt};
            if (std::regex_search(argString, matchFlagShort) ||
                    std::regex_search(argString, matchFlagLong)) {
                parsedArgs[arg] =
                    ""; // empty string for flags, will just be checked using .contains
            }
        };
    }

    if (missingRequired) {
        printHelp(argv[0]);
        exit(ExitCode::ERROR);
    }
    return parsedArgs;
}

} // end namespace ArgParse

// taken from https://stackoverflow.com/a/17708801
string url_encode(const string &value) {
    std::ostringstream escaped;
    escaped.fill('0');
    escaped << std::hex;

    for (string::const_iterator i = value.begin(), n = value.end(); i != n;
         ++i) {
        string::value_type c = (*i);

        // Keep alphanumeric and other accepted characters intact
        if (isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~') {
            escaped << c;
            continue;
        }

        // Any other characters are percent-encoded
        escaped << std::uppercase;
        escaped << '%' << std::setw(2) << int((unsigned char)c);
        escaped << std::nouppercase;
    }

    return escaped.str();
}

