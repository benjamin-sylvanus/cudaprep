//
// Created by Benjamin Sylvanus on 3/8/23.
//

#include "viewer.h"
#include <iostream>
#include <map>
#include <cstring>
#include "string.h"
#include <iostream>
#include <map>
#include <string>
#include <string_view>
#include <sys/ioctl.h>
#include <stdio.h>
#include <unistd.h>

using std::cout;
using std::endl;

/**
*
Name            FG  BG
Black           30  40
Red             31  41
Green           32  42
Yellow          33  43
Blue            34  44
Magenta         35  45
Cyan            36  46
White           37  47
Bright Black    90  100
Bright Red      91  101
Bright Green    92  102
Bright Yellow   93  103
Bright Blue     94  104
Bright Magenta  95  105
Bright Cyan     96  106
Bright White    97  107
*/

std::map<std::string, std::string> colors{
        {"p",  "\x1B[95m"},
        {"r",  "\x1B[91m"},
        {"b",  "\x1B[34m"},
        {"g",  "\x1B[32m"},
        {"y",  "\x1B[93m"},
        {"w",  "\033[0m"},
        {"bb", "\x1B[94m"}
};

struct winsize w;


void print_map(std::map<std::string, std::string> m, int useColor) {
    // show content:
    for (std::map<std::string, std::string>::iterator it = m.begin(); it != m.end(); ++it)

        (useColor) ? printf("%s%s%s ---> %s%s%s\n", colors["g"].c_str(),
                                it->first.c_str(), colors["w"].c_str(),
                                colors["b"].c_str(),it->second.c_str(),
                                colors["w"].c_str()) :
                                printf("%s ---> %s\n", it->first.c_str(),
                                it->second.c_str());
}

void viewer::display(int option) {

}

void viewer::show(simulation sim) {
    ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);
    this->str.assign(w.ws_col, '-');
    std::string temp = this->str;
    std::string value = "Configuration";
    temp.replace((temp.length() / 2) - (value.length() / 2), value.length(), value);
    (this->useColor) ? cout << colors["p"] << this->str << endl
                      << temp << endl << this->str << colors["w"] << endl :
                      cout << this->str << endl << temp
                      << endl << this-> str << endl;
    cout << "Particle_num: " << sim.getParticle_num() << endl;
    cout << "Step_num: " << sim.getStep_num() << endl;
    cout << "Vsize: " << sim.getVsize() << endl;
    cout << "Scale: " << sim.getScale() << endl;
    cout << "Init_in: " << sim.getInit_in() << endl;
    cout << "Step_size: " << sim.getStep_size() << endl;
    cout << "Perm_prob: " << sim.getPerm_prob() << endl;
    cout << "D0: " << sim.getD0() << endl;
    cout << "D: " << sim.getD() << endl;
    cout << "Tstep: " << sim.getTstep() << endl;
    cout << "Path: " << sim.getResultPath() << endl;
    cout << "Save All" << sim.getSaveAll() << endl; 

}

void viewer::welcome() {

    ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);
    this->str.assign(w.ws_col, '-');
    std::string temp = this->str;
    std::string value = "Welcome";
    temp.replace((temp.length() / 2) - (value.length() / 2), value.length(), value);

    (this->useColor) ? cout << colors["bb"] << this->str << endl << temp
                            << endl << this->str << colors["w"] << endl :
                            cout << this->str << endl << temp
                            << endl << this->str << endl;
    showCommands();
}
viewer::viewer()
{
  ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);
  this->command = "-h help\n-c commands\n-s show configuration\n-d <arg> show argument\n-<arg> <value> set argument to value\n-args show arguments";
  this->help = "-c -a will show options for input.\nArguments require a parameter:\n\t<-np 10000> sets the particle number to 10000\n\nInputs can be chained:\n\t<-ns 2000 -np 1000> sets the particle number to 1000 and step number to 2000.\n\nFor additional information checkout the ReadMe or documentation on github.";
  this->str.assign(w.ws_col, '-');
  this->useColor = 1;
}

viewer::viewer(int c) {
    ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);
    this->command = "-h help\n-c commands\n-s show configuration\n-d <arg> show argument\n-<arg> <value> set argument to value\n-args show arguments";
    this->help = "-c -a will show options for input.\nArguments require a parameter:\n\t<-np 10000> sets the particle number to 10000\n\nInputs can be chained:\n\t<-ns 2000 -np 1000> sets the particle number to 1000 and step number to 2000.\n\nFor additional information checkout the ReadMe or documentation on github.";
    this->str.assign(w.ws_col, '-');
    this->useColor = c;
}


void viewer::showargs(std::map<std::string, std::string> args) {
    ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);
    this->str.assign(w.ws_col, '-');
    std::string temp = this->str;
    std::string value = "Args";
    temp.replace((temp.length() / 2) - (value.length() / 2), value.length(), value);
    (this->useColor) ? cout << colors["p"] << this->str << endl << temp << endl << this->str << colors["w"] << endl : cout << this->str << endl << temp << endl << this->str << endl;
    print_map(args, this->useColor);
}

void viewer::showCommands() {
    ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);
    this->str.assign(w.ws_col, '-');
    std::string temp = this->str;
    std::string value = "Commands";
    temp.replace((temp.length() / 2) - (value.length() / 2), value.length(), value);
    (this->useColor) ? cout << colors["p"] << this->str << endl << temp << endl << this->str << colors["w"] << endl :
    cout << this->str << endl << temp << endl << this->str << endl;
    cout << command << endl;
}

void viewer::showHelp() {
    //
    ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);
    this->str.assign(w.ws_col, '-');
    std::string temp = this->str;
    std::string value = "Help";
    temp.replace((temp.length() / 2) - (value.length() / 2), value.length(), value);
    (this->useColor) ? cout << colors["p"] << this->str << endl << temp << endl << this->str << colors["w"] << endl :
    cout << this->str << endl << temp << endl << this->str << endl;
    cout << help << endl;
}

void viewer::AlertNoParameter(std::string target)
{
    // Use red to alert
    (this->useColor) ? cout << colors["y"] << str << colors["r"] << target << " requires additional parameters."
     << endl << colors["y"] << str << endl << colors["w"] << endl :
    cout << str << target << " requires additional parameters." << endl << str << endl << endl;

}
