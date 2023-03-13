//
// Created by Benjamin Sylvanus on 3/8/23.
//

#include "controller.h"
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include "simulation.h"
#include "simreader.h"
#include <string.h>
#include <vector>
#include <algorithm>
#include <map>
using namespace std;




// controller::controller(simulation& sim) : sim(sim) {
//     this->view = viewer();
// }

// for string delimiter


std::vector<std::string> split(std::string s, std::string delimiter) {
    size_t pos_start = 0, pos_end, delim_len = delimiter.length();
    std::string token;
    std::vector<std::string> res;

    while ((pos_end = s.find(delimiter, pos_start)) != std::string::npos) {
        token = s.substr (pos_start, pos_end - pos_start);
        pos_start = pos_end + delim_len;
        res.push_back (token);
    }

    res.push_back (s.substr (pos_start));
    return res;
}

controller::controller(std::string path)
{
  simreader reader(&path);
  simulation sim(reader);
  this->sim = sim;
  this->view = viewer();
  this->commands = {"h","c","a"};

  std::map<std::string, Controls> map{
  {"c", Controls::Commands},
  {"h", Controls::Help},
  {"a", Controls::Args},
  {"s", Controls::Show},
  {"ss",Controls::StepSize},
  {"pp",Controls::PermeationProbability},
  {"d0",Controls::IntrinsicDiffusivity},
  {"d", Controls::Distance},
  {"ts",Controls::TimeStep},
  {"sc",Controls::Scale},
  {"vs",Controls::VoxelSize},
  {"ns",Controls::NStep},
  {"np",Controls::NPar}};

  std::map<std::string, std::string> targets{
  {"c", "Controls::Commands"},
  {"h", "Controls::Help"},
  {"a", "Controls::Args"},
  {"s", "Controls::Show"},
  {"ss","Controls::StepSize"},
  {"pp","Controls::PermeationProbability"},
  {"d0","Controls::IntrinsicDiffusivity"},
  {"d", "Controls::Distance"},
  {"ts","Controls::TimeStep"},
  {"sc","Controls::Scale"},
  {"vs","Controls::VoxelSize"},
  {"ns","Controls::NStep"},
  {"np","Controls::NPar"}};


  this->map = map;
  this->targets = targets;

  // {ss,pp,d0,d,ts,pd,ad,s,vs};

  this->args = {"ss", "pp", "d0", "d", "ts", "pd", "ad", "s", "vs", "ns","np"};
  /**
  *
      simreader reader;
      std::vector<double> swc; // swc array;
      std::vector<std::uint64_t> lut; // lut of simulation
      std::vector<std::uint64_t> index; // index array
      std::vector<std::uint64_t> pairs; // pairs of swc
      std::vector<std::uint64_t> bounds; // bounds of geometry
      double particle_num; // number of particles
      double step_num; // number of steps to simulate
      double init_in; // initialize particles inside?

      step_size
      perm_prob
      D0
      d
      tstep
      parameterdata
      arraydims
      scale
      vsize
  */
}

void controller::handleargument(std::string argument, std::string value)
{
  bool valid=false;

  if (! std::count(this->args.begin(), this->args.end(), argument))
  {
    // printf("invalid argument\n\n");
    // view.showargs(this->args);
  }

}

void controller::handlecommand(std::vector<string> sub)
{
    std::string c = sub[0];

    if(this->map.count(c))
    {
    Controls input = this->map[c];
    std::string target = this->targets[c];

    switch(input)
    {
      case Controls::Help:
        view.showHelp();
        break;

      case Controls::Args:
        this->view.showargs(this->targets);
        break;

      case Controls::Commands:
        view.showCommands();
        break;
      case Controls::Show:
        view.show(this->sim);
        break;
        
      case Controls::NStep:
        (sub.size() < 2) ? view.AlertNoParameter(target) : this->sim.setStep_num(std::stod(sub[1]));
        break;

      case Controls::NPar:
        (sub.size() < 2) ? view.AlertNoParameter(target) : this->sim.setParticle_num(std::stod(sub[1]));
        break;

      case Controls::StepSize:
        // validate additional parameters;
        (sub.size() < 2) ? view.AlertNoParameter(target) : this->sim.setStep_size(std::stod(sub[1]));
        break;

      case Controls::PermeationProbability:
        //validate additional parameters;
        (sub.size() < 2) ? view.AlertNoParameter(target) : this->sim.setPerm_prob(std::stod(sub[1]));
        break;

      case Controls::IntrinsicDiffusivity:
        //validate additional parameters;
        (sub.size() < 2) ? view.AlertNoParameter(target) : this->sim.setD0(std::stod(sub[1]));
        break;

      case Controls::Distance:
        //validate additional parameters;
        (sub.size() < 2) ? view.AlertNoParameter(target) : this->sim.setD(std::stod(sub[1]));
        break;

      case Controls::TimeStep:
        //validate additional parameters;
        (sub.size() < 2) ? view.AlertNoParameter(target) : this->sim.setTstep(std::stod(sub[1]));
        break;

      case Controls::Scale:
        //validate additional parameters;
        (sub.size() < 2) ? view.AlertNoParameter(target) : this->sim.setScale(std::stod(sub[1]));
        break;

      case Controls::VoxelSize:
        //validate additional parameters;
        (sub.size() < 2) ? view.AlertNoParameter(target) : this->sim.setVsize(std::stod(sub[1]));
        break;


      default:
        printf("Invalid Argument\n");
        break;


    }

  }

  else
  {
    printf("Invalid Argument");
  }
}

void controller::handleinput(std::string buf)
{
  std::vector<std::string> elements = split(buf,"-");

  for(const string& s: elements)
  {
      if (s.length() > 0)
      {
        std::vector<string> sub = split(s," ");
        handlecommand(sub);
      }
  }
}

void controller::start() {
    view.welcome();

    while (true)
    {
        string buf;
        getline(cin,buf);
        bool b = (buf == "q" || buf == "Q" || buf == "done");
        if(b)
        {
           break;
        }else
        {
          handleinput(buf);
        }


    }
}

simulation controller::getSim()
{
  return this->sim;
}
