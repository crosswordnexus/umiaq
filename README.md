# umiaq
A pattern-matching tool to run against a word list

Word list is from https://www.spreadthewordlist.com/, CC BY-NC-SA 4.0

# Samples

## Simple pattern matching
```
$ python umiaq.py "l.....x" -n 5
LANDTAX
LEXUSRX
LILJINX
LOCKBOX
Total time: 0.389 seconds

$ python umiaq.py "..i[sz]e" -n 5
ARISE
GUISE
LUISE
MAIZE
NOISE
Total time: 0.333 seconds

$ python umiaq.py "#@#@#@#@#@#@#@" -n 5
COMEFACETOFACE
DELIBERATIVELY
MILITARYPOLICE
MINUTEBYMINUTE
NATURALABILITY
Total time: 0.302 seconds

$ python umiaq.py "*xj*" -n 5
JAGUARXJ
FLEXJOBS
ORTHODOXJEW
FIXITFELIXJR
Total time: 0.347 seconds
```  

## Matching using variables
```
$ python umiaq.py "AA" -n 5
GAGA
KOKO
MAMA
PAPA
YOYO
Total time: 0.459 seconds

$ python umiaq.py "AB;BA;|A|=1" -n 5
APE • PEA
BRO • ROB
CAM • AMC
DAD • ADD
EAR • ARE
Total time: 7.332 seconds

$ python umiaq.py "AkB;AlB" -n 5
SKY • SLY
BAKE • BALE
BIKE • BILE
COKE • COLE
MIKE • MILE
Total time: 2.790 seconds

$ python umiaq.py "A###B;A@@@B" -n 5
SCHWA • SEEYA
HDQRS • HAYES
RSVPS • ROUES
BANDTS • BAYOUS
PICKLED • PIEEYED
Total time: 1.355 seconds
```  
