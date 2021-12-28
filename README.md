# umiaq
A pattern-matching tool to run against a word list

Samples

    $ python umiaq.py "*...AredA...*"
    CATCHREDHANDED
    MICHELINSTARREDRESTAURANT
    MICHELINSTARREDRESTAURANTS
    CLEAREDAPATH
    Total time: 0.224 seconds

    $ python umiaq.py "AB;BA"
    [...]
    SOSO • SOSO
    SOWN • OWNS
    OWED • WEDO
    DODO • DODO
    Maximum number of outputs reached
    Total time: 5.971 seconds

    $ python umiaq.py "A;Ared"
    [...]
    DESI • DESIRED
    MARTY • MARTYRED
    CHAI • CHAIRED
    BUTTE • BUTTERED
    MOO • MOORED
    Total time: 1.611 seconds
    
    $ python umiaq.py "AB;BC;CD"
    [...]
    CABLE • BLEB • BE
    WILDE • DEB • BE
    GIDE • DEB • BE
    CHORAL • ORALB • BE
    Maximum number of outputs reached
    Total time: 5.824 seconds
    
