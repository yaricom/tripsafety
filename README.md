# tripsafety
## Overwiev
GDL is responsible for scheduling on-going deliveries to customers. Continuously improving safety is always an 
important objective for GDL. Because safety has been the top priority at GDL, accidents during delivery are extremely 
rare. In the effort to continuously improve safety even where there are no accidents, GDL uses on board computers (OBC) 
to monitor properties like speed, acceleration, deceleration, and stability during deliveries and to identify events 
which might indicate increased risk. For example, a deceleration and/or stability event can indicate that pilots we 
required to take evasive action, perhaps to avoid a collision or other hazard.

GDL is searching for ways to use its information about past trips and events to help schedule ever safer trips by 
choosing routes that avoid these kinds of events.

It was done as part of crowdsourcing contest on TopCoder: [Contest: Trip Safety ](https://community.topcoder.com/longcontest/?module=ViewProblemStatement&rd=16307&pm=13624)

### Special Conditions
Additional requirement was to create crafted algorithm which may be compilled and runned in the AWS runner.
Limits:
-	The time limit is 5 minutes. The memory limit is 2048 megabytes.
-	The compilation time limit is 30 seconds. You can find information about compilers that we use and compilation options here.

The main algorithm runner implemented in: [TripSafetyFactors.h](https://github.com/yaricom/tripsafety/blob/master/TripSafety/TripSafety/TripSafetyFactors.h)
