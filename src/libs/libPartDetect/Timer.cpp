#include <sys/times.h>

#include "Timer.h"

Timer *Timer::m_pTimer = 0;

Timer::Timer() :m_last_tms_utime(0), m_last_tms_stime(0) 
{}

Timer *Timer::getTimer() 
{
  if (m_pTimer == 0) 
    m_pTimer = new Timer();

  return m_pTimer;
}

void Timer::getElapsedTime(clock_t &user, clock_t& system)
{
  tms _tms;
  times(&_tms);

  user = _tms.tms_utime - m_last_tms_utime;
  system = _tms.tms_stime - m_last_tms_stime;

  m_last_tms_utime = _tms.tms_utime;
  m_last_tms_stime = _tms.tms_stime;
}
