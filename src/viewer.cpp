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

std::map <std::string,std::string> colors {
{"p","\x1B[95m"},
{"r","\x1B[91m"},
{"b","\x1B[34m"},
{"g","\x1B[32m"},
{"y","\x1B[93m"},
{"w","\033[0m"},
{"bb","\x1B[94m"}
};

struct winsize w;



void print_map(std::map<std::string, std::string> m)
{
  // show content:
  for (std::map<std::string,std::string>::iterator it=m.begin(); it!=m.end(); ++it)
    printf("%s%s%s ---> %s%s%s\n",colors["g"].c_str(),it->first.c_str(),colors["w"].c_str(),colors["b"].c_str(),it->second.c_str(),colors["w"].c_str());
}

void viewer::display(int option) {

}

void viewer::show(simulation sim) {
  ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);
  this->str.assign(w.ws_col,'-');
  std::string temp = this->str;
  std::string value = "Configuration";
  temp.replace((temp.length()/2) - (value.length()/2),value.length(),value);
  cout << colors["p"] <<  this->str << endl << temp << endl << this->str <<colors["w"] <<endl;
  cout<< "Particle_num: " <<sim.getParticle_num()<< endl;
  cout<< "Step_num: " << sim.getStep_num() << endl;
  cout<< "Vsize: " << sim.getVsize() << endl;
  cout<< "Scale: " << sim.getScale() << endl;
  cout<< "Init_in: " << sim.getInit_in() << endl;
  cout<< "Step_size: " << sim.getStep_size() << endl;
  cout<< "Perm_prob: " << sim.getPerm_prob() << endl;
  cout<< "D0: " << sim.getD0() << endl;
  cout<< "D: " << sim.getD() << endl;
  cout<< "Tstep: " << sim.getTstep() << endl;

  // std::vector<double> simulation::getSwc()
  //
  // std::vector <std::uint64_t> simulation::getLut()
  //
  // std::vector <std::uint64_t> simulation::getIndex()
  //
  // std::vector <std::uint64_t> simulation::getPairs()
  //
  // std::vector <std::uint64_t> simulation::getbounds()
  //
  // std::vector<double> simulation::getParameterdata()
  //
  // std::vector <std::vector<uint64_t>> simulation::getArraydims()
}

void viewer::welcome() {

    ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);
    this->str.assign(w.ws_col,'-');
    std::string temp = this->str;
    std::string value = "Welcome";
    temp.replace((temp.length()/2) - (value.length()/2),value.length(),value);
    cout << colors["bb"] <<  this->str<< endl << temp << endl << this->str <<colors["w"] <<endl;
    showCommands();
}

viewer::viewer()
{
    ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);
    this->command = "-h help\n-c commands\n-s show configuration\n-d <arg> show argument\n-<arg> <value> set argument to value\n-args show arguments";
    this->str.assign(w.ws_col,'-');
}


void viewer::showargs(std::map<std::string,std::string> args)
{
  ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);
  this->str.assign(w.ws_col,'-');
  std::string temp = this->str;
  std::string value = "Args";
  temp.replace((temp.length()/2) - (value.length()/2),value.length(),value);
  cout << colors["p"] <<  this->str<< endl << temp << endl << this->str <<colors["w"] <<endl;
  print_map(args);
}

void viewer::showCommands()
{
  ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);
  this->str.assign(w.ws_col,'-');
  std::string temp = this->str;
  std::string value = "Commands";
  temp.replace((temp.length()/2) - (value.length()/2),value.length(),value);
  cout << colors["p"] <<  this->str<< endl << temp << endl << this->str <<colors["w"] <<endl;
  cout << command << endl;
}

void viewer::showHelp()
{
  //
  ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);
  this->str.assign(w.ws_col,'-');
  cout<<help<<endl;

}

void viewer::AlertNoParameter(std::string target)
{
  //Use red to alert
  cout<< colors["y"]<<str<<colors["r"]<<target<< " requires additional parameters."<<endl<<colors["y"]<<str<<endl<<colors["w"]<<endl;
}
