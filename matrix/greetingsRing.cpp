/* greetingsRing.c -- greetings program
 *
 * Send a message from one process to the next rank 
 *	process, starting and ending at rank 0.
 *
 * Input: none.
 * Output: contents of messages received by process 0.
 *
 * See Chapter 3, pp. 41 & ff in PPMPI.
 */
#include <stdio.h>
#include <string.h>
#include <mpi.h>

main(int argc, char* argv[]) {
    int         my_rank;       /* rank of process      */
    int         p;             /* number of processes  */
    int         source;        /* rank of sender       */
    int         dest;          /* rank of receiver     */
    int         tag = 0;       /* tag for messages     */
    char        message[100];  /* storage for message  */
    MPI_Status  status;        /* return status for    */
                               /* receive              */

    /* Start up MPI */
    MPI_Init(&argc, &argv);

    /* Find out process rank  */
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    /* Find out number of processes */
    MPI_Comm_size(MPI_COMM_WORLD, &p);

	if (my_rank < p - 1) {
		dest = my_rank + 1;
	} else {
		dest = 0;
	}
	
    if (my_rank == 0) {
	printf("Process 0: Initiating greetings message\n");
        sprintf(message, "Greetings from process %d!", my_rank);
        /* Use strlen+1 so that '\0' gets transmitted */
        MPI_Send(message, strlen(message)+1, MPI_CHAR, 
            dest, tag, MPI_COMM_WORLD);
		source = p - 1;
        MPI_Recv(message, 100, MPI_CHAR, source, tag, 
            MPI_COMM_WORLD, &status);
        printf("Process %d: %s\n", my_rank, message);
    } else { 
	source = my_rank - 1;
        MPI_Recv(message, 100, MPI_CHAR, source, tag, 
            MPI_COMM_WORLD, &status);
        printf("Process %d: %s\n", my_rank, message);
        sprintf(message, "Greetings from process %d!", my_rank);
	MPI_Send(message, strlen(message)+1, MPI_CHAR, 
            dest, tag, MPI_COMM_WORLD);
    }

    /* Shut down MPI */
    MPI_Finalize();
} /* main */
