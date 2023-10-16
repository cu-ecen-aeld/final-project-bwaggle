#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <unistd.h>
#include <sys/socket.h>
#include <arpa/inet.h>

// #define FILE_PATH "/home/bwaggle/final-project/data/logs/"
#define FILE_PATH "/home/app/data/photos/"
// #define SERVER_IP "127.0.0.1"
#define SERVER_IP "192.168.86.33"
#define SERVER_PORT 9000

int create_client_socket() {
    int client_socket = socket(AF_INET, SOCK_STREAM, 0);
    if (client_socket == -1) {
        perror("Error in socket creation");
        exit(1);
    }
    return client_socket;
}

int connect_to_server(int client_socket, const char* server_ip, int server_port) {
    struct sockaddr_in server_addr;
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(server_port);
    server_addr.sin_addr.s_addr = inet_addr(server_ip);

    int connection_status = connect(client_socket, (struct sockaddr *)&server_addr, sizeof(server_addr));
    if (connection_status == -1) {
        perror("Error in connection");
        close(client_socket);
        exit(1);
    }

    return connection_status;
}

void send_file(int client_socket, const char* file_path) {
    FILE *file = fopen(file_path, "rb");
    if (file == NULL) {
        perror("Error opening file");
        return;
    }

    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    rewind(file);

    char *file_buffer = (char *)malloc(file_size + 5);
    if (file_buffer == NULL) {
        perror("Error allocating memory");
        fclose(file);
        return;
    }

    fread(file_buffer, 1, file_size, file);

    // Send end of packet delimeter
    strcpy(file_buffer + file_size, "EOF\n");

    // Send the file data
    send(client_socket, file_buffer, file_size + 5, 0);
    remove(file_path);
    printf("Removing: %s\n", file_path);
    free(file_buffer);
}

int main() {
    int client_socket;
    int connection_status;

    client_socket = create_client_socket();
    connection_status = connect_to_server(client_socket, SERVER_IP, SERVER_PORT);

    while (1) {
        DIR *dir;
        struct dirent *entry;

        dir = opendir(FILE_PATH);
        if (dir == NULL) {
            perror("Error opening directory");
            exit(1);
        }

        while ((entry = readdir(dir)) != NULL) {
            
            if (entry->d_type == 8) {
                printf("Found entry %s, %d\n", entry->d_name, entry->d_type);
                if (strstr(entry->d_name, ".jpg") != NULL) {
                    char filepath[512];
                    snprintf(filepath, sizeof(filepath), "%s%s", FILE_PATH, entry->d_name);
                    send_file(client_socket, filepath);
                }
            }
        }
        closedir(dir);

        sleep(1);
    }

    close(client_socket);

    return 0;
}
