
CC ?= $(CROSS_COMPILE)gcc
# OBJFILES ?= aesdsocket.o
OBJFILES ?= face_server.o
# MAIN ?= aesdsocket
MAIN ?= face_server
CFLAGS ?= -g -Wall -Werror
LDFLAGS ?= -pthread

all: $(MAIN)

$(MAIN): $(OBJFILES)
	$(CC) -o $(MAIN) $(OBJFILES) $(CFLAGS) $(LDFLAGS)

%.o: %.c 
	$(CC) -c -o $@ $<

clean:
	rm -f $(OBJFILES) $(MAIN)