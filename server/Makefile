
CC ?= $(CROSS_COMPILE)gcc
OBJFILES ?= aesdsocket.o
MAIN ?= aesdsocket
CFLAGS ?= -g -Wall -Werror
LDFLAGS ?= -pthread

all: $(MAIN)

$(MAIN): $(OBJFILES)
	$(CC) -o $(MAIN) $(OBJFILES) $(CFLAGS) $(LDFLAGS)

%.o: %.c 
	$(CC) -c -o $@ $<

clean:
	rm -f $(OBJFILES) $(MAIN)