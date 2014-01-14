#ifndef _TIMER_H
#define TIMER_H
#include <sys/times.h>

class Timer {
 public:
  static Timer* getTimer();

  // returns user/system time elapsed since the last call
  void getElapsedTime(clock_t &user, clock_t &system);

 protected:
  Timer();

 private:
  static Timer *m_pTimer;

  clock_t m_last_tms_utime;
  clock_t m_last_tms_stime;  
};

#endif
