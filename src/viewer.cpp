//
// Created by Benjamin Sylvanus on 3/8/23.
//

#include "viewer.h"

void viewer::display(int option) {

}

void viewer::show() {

}

void viewer::welcome() {
    printf("-------------------------------------------\n");
    printf("--------------------Welcome----------------\n");
    printf("-------------------------------------------\n");
    printf("%s\n",command.c_str());
}

viewer::viewer()
{
    system("clear");
    this->command = "-h help\n-c commands\n-s show configuration-d <arg> show argument\n-<arg> <value> set argument to value\n-args show arguments";
}

void viewer::showargs(std::vector<std::string> args)
{
  printf("-------------------------------------------\n");
  printf("--------------------Args-------------------\n");
  printf("-------------------------------------------\n");
  for(std::string s: args)
  {
    printf("%s\n",s.c_str());
  }
}

void viewer::AlertNoParameter(std::string target)
{
  //Use red to alert
  printf("%s requires additional parameters.\n",target.c_str());
  printf("Please try again\n");
  printf("-------------------------------------------\n");

}
