#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>

#define BILLION 1E9

void swap (float *a, float *b){
	float tmp = *a;
	*a = *b;
	*b = tmp;
}

int main(int argc, char* argv[])
{
	int rank, proc_size, num_size, list_size, head_gindex, tail_gindex, remainder, rc;
	int isOneMore = 0;
	MPI_File fh;
	MPI_Status st;
	MPI_Comm valid_comm, other_comm;
	struct timespec IO_start, IO_end, CPU_start, CPU_end, Comm_start, Comm_end;
	double IO_time, IO_time_temp, CPU_time, CPU_time_temp, Comm_time, Comm_time_temp;
	IO_time = IO_time_temp = CPU_time = CPU_time_temp = Comm_time = Comm_time_temp = 0;
	num_size = atoi(argv[1]);

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &proc_size);
	if(proc_size > num_size) proc_size = num_size;
	if(rank >= proc_size){
		MPI_Comm_split(MPI_COMM_WORLD, rank / proc_size, rank, &other_comm);
		MPI_Comm_free(&other_comm);
		MPI_Finalize();
		return 0;
	}
	else MPI_Comm_split(MPI_COMM_WORLD, rank / proc_size, rank, &valid_comm);

	isOneMore = rank < (num_size % proc_size);
	remainder = num_size % proc_size;
	if(isOneMore) {
		list_size = num_size / proc_size + 1;
		head_gindex = list_size * rank;
	}
	else {
		list_size = num_size / proc_size;
		head_gindex = (list_size+1)*remainder + list_size*(rank-remainder);
	}
	tail_gindex = head_gindex + list_size - 1;
	//printf("Rank: %d, list_size: %d\n", rank, list_size);
	float* list = malloc(sizeof(float)*list_size);

	clock_gettime(CLOCK_MONOTONIC, &IO_start);
	rc = MPI_File_open(valid_comm, argv[2], MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
	MPI_File_read_at(fh, sizeof(float)*head_gindex, list, list_size, MPI_FLOAT, &st);
	MPI_File_close(&fh);
	clock_gettime(CLOCK_MONOTONIC, &IO_end);
	IO_time_temp += (IO_end.tv_sec - IO_start.tv_sec) + (IO_end.tv_nsec-IO_start.tv_nsec)/BILLION;
	//printf("Rank: %d, IO_read: %lf\n", rank, IO_time_temp);
	//End MPI IO
	
	int swap_type = (head_gindex%2 == 0)*2 + (tail_gindex%2 == 0)*1;
	int swap_time = 0, i;
	float exchange_head, exchange_tail;
	MPI_Request req;
	//printf("Rank: %d, swap_type: %d\n", rank, swap_type);
	while(!swap_time) {
		swap_time = 1;
		switch(swap_type) {
			case 0: // no rank = 0 case. In even phase, no worry about rank = list_size-1
				clock_gettime(CLOCK_MONOTONIC, &Comm_start);
	            MPI_Isend(&list[0], 1, MPI_FLOAT, rank-1, rank, valid_comm, &req);
				clock_gettime(CLOCK_MONOTONIC, &Comm_end);
                Comm_time_temp += (Comm_end.tv_sec - Comm_start.tv_sec) + (Comm_end.tv_nsec-Comm_start.tv_nsec)/BILLION;				

				clock_gettime(CLOCK_MONOTONIC, &CPU_start);
				for(i=1; i+1<list_size; i+=2) {
	                    if(list[i] > list[i+1]) { swap(&list[i], &list[i+1]); swap_time = 0; }
	            }
				clock_gettime(CLOCK_MONOTONIC, &CPU_end);
			    CPU_time_temp += (CPU_end.tv_sec - CPU_start.tv_sec) + (CPU_end.tv_nsec-CPU_start.tv_nsec)/BILLION;
		
				clock_gettime(CLOCK_MONOTONIC, &Comm_start);	
				MPI_Recv(&exchange_head, 1, MPI_FLOAT, rank-1, rank-1, valid_comm, &st);
				clock_gettime(CLOCK_MONOTONIC, &Comm_end);
               	Comm_time_temp += (Comm_end.tv_sec - Comm_start.tv_sec) + (Comm_end.tv_nsec-Comm_start.tv_nsec)/BILLION;
			
				if(list[0] < exchange_head) { list[0] = exchange_head; swap_time = 0; }
				break;
	
			case 1:	//no rank = 0 case
				if(rank < proc_size-1) {	//not the last, exchange both head and tail with previous and next process
					clock_gettime(CLOCK_MONOTONIC, &Comm_start);
					MPI_Isend(&list[list_size-1], 1, MPI_FLOAT, rank+1, rank, valid_comm, &req);
	                MPI_Isend(&list[0], 1, MPI_FLOAT, rank-1, rank, valid_comm, &req);
					clock_gettime(CLOCK_MONOTONIC, &Comm_end);
					Comm_time_temp += (Comm_end.tv_sec - Comm_start.tv_sec) + (Comm_end.tv_nsec-Comm_start.tv_nsec)/BILLION;
					
					clock_gettime(CLOCK_MONOTONIC, &CPU_start);
					for(i=1; i+1<list_size-1; i+=2) { 
						if(list[i] > list[i+1]) { swap(&list[i], &list[i+1]); swap_time = 0; }
					}
					clock_gettime(CLOCK_MONOTONIC, &CPU_end);
                	CPU_time_temp += (CPU_end.tv_sec - CPU_start.tv_sec) + (CPU_end.tv_nsec-CPU_start.tv_nsec)/BILLION;
	
					clock_gettime(CLOCK_MONOTONIC, &Comm_start);
					MPI_Recv(&exchange_tail, 1, MPI_FLOAT, rank+1, rank+1, valid_comm, &st);
					MPI_Recv(&exchange_head, 1, MPI_FLOAT, rank-1, rank-1, valid_comm, &st);
					clock_gettime(CLOCK_MONOTONIC, &Comm_end);
					Comm_time_temp += (Comm_end.tv_sec - Comm_start.tv_sec) + (Comm_end.tv_nsec-Comm_start.tv_nsec)/BILLION;
	                //printf("exchange_tail: %f\n", exchange_tail);
	                //
					if(list[list_size-1] > exchange_tail) { list[list_size-1] = exchange_tail; swap_time = 0; }
					if(list[0] < exchange_head) { list[0] = exchange_head; swap_time = 0; }
	            }
				if(rank ==  proc_size-1) {	//the last, only exchange head with previous process
					clock_gettime(CLOCK_MONOTONIC, &Comm_start);
					MPI_Isend(&list[0], 1, MPI_FLOAT, rank-1, rank, valid_comm, &req);
					clock_gettime(CLOCK_MONOTONIC, &Comm_end);
					Comm_time_temp += (Comm_end.tv_sec - Comm_start.tv_sec) + (Comm_end.tv_nsec-Comm_start.tv_nsec)/BILLION;

					clock_gettime(CLOCK_MONOTONIC, &CPU_start);
					for(i=1; i+1<list_size-1; i+=2) {
	                    if(list[i] > list[i+1]) { swap(&list[i], &list[i+1]); swap_time = 0; }
	                }
					clock_gettime(CLOCK_MONOTONIC, &CPU_end);
                    CPU_time_temp += (CPU_end.tv_sec - CPU_start.tv_sec) + (CPU_end.tv_nsec-CPU_start.tv_nsec)/BILLION;
		
					clock_gettime(CLOCK_MONOTONIC, &Comm_start);
					MPI_Recv(&exchange_head, 1, MPI_FLOAT, rank-1, rank-1, valid_comm, &st);
					clock_gettime(CLOCK_MONOTONIC, &Comm_end);
					Comm_time_temp += (Comm_end.tv_sec - Comm_start.tv_sec) + (Comm_end.tv_nsec-Comm_start.tv_nsec)/BILLION;

					if(list[0] < exchange_head) { list[0] = exchange_head; swap_time = 0; }
				}
				break;

			case 2: //In even phase, no worry about rank = 0, rank = list_size -1
				clock_gettime(CLOCK_MONOTONIC, &CPU_start);
				for(i=0; i+1<list_size; i+=2) {
	                    if(list[i] > list[i+1]) { swap(&list[i], &list[i+1]); swap_time = 0; }
	            }
				clock_gettime(CLOCK_MONOTONIC, &CPU_end);
                CPU_time_temp += (CPU_end.tv_sec - CPU_start.tv_sec) + (CPU_end.tv_nsec-CPU_start.tv_nsec)/BILLION;
				break;

			case 3: //In even phase, no worry about rank = 0 case
				if(rank < proc_size-1) { //not the last, only exchange tail with next process
					if(proc_size > 1) {
						clock_gettime(CLOCK_MONOTONIC, &Comm_start);
						MPI_Isend(&list[list_size-1], 1, MPI_FLOAT, rank+1, rank, valid_comm, &req);
						clock_gettime(CLOCK_MONOTONIC, &Comm_end);
						Comm_time_temp += (Comm_end.tv_sec - Comm_start.tv_sec) + (Comm_end.tv_nsec-Comm_start.tv_nsec)/BILLION;
					}
					
					clock_gettime(CLOCK_MONOTONIC, &CPU_start);
					for(i=0; i+1<list_size-1; i+=2) {
	                    if(list[i] > list[i+1]) { swap(&list[i], &list[i+1]); swap_time = 0; }
	                }
					clock_gettime(CLOCK_MONOTONIC, &CPU_end);
	                CPU_time_temp += (CPU_end.tv_sec - CPU_start.tv_sec) + (CPU_end.tv_nsec-CPU_start.tv_nsec)/BILLION;
					
					if(proc_size > 1) {
						clock_gettime(CLOCK_MONOTONIC, &Comm_start);
						MPI_Recv(&exchange_tail, 1, MPI_FLOAT, rank+1, rank+1, valid_comm, &st);
						clock_gettime(CLOCK_MONOTONIC, &Comm_end);
						Comm_time_temp += (Comm_end.tv_sec - Comm_start.tv_sec) + (Comm_end.tv_nsec-Comm_start.tv_nsec)/BILLION;
						if(list[list_size-1] > exchange_tail) { list[list_size-1] = exchange_tail; swap_time = 0; }
					}
				}
				if(rank == proc_size-1) { //the last, only local swap
					clock_gettime(CLOCK_MONOTONIC, &CPU_start);
					for(i=0; i+1<list_size-1; i+=2) {
	                    if(list[i] > list[i+1]) { swap(&list[i], &list[i+1]); swap_time = 0; }
	                }
					clock_gettime(CLOCK_MONOTONIC, &CPU_end);
                    CPU_time_temp += (CPU_end.tv_sec - CPU_start.tv_sec) + (CPU_end.tv_nsec-CPU_start.tv_nsec)/BILLION;
				}
		}
		MPI_Barrier(valid_comm);	
	
		switch(swap_type) {
			case 0:
				if(rank < proc_size-1) {
					clock_gettime(CLOCK_MONOTONIC, &Comm_start);
					MPI_Isend(&list[list_size-1], 1, MPI_FLOAT, rank+1, rank, valid_comm, &req);
					clock_gettime(CLOCK_MONOTONIC, &Comm_end);
                    Comm_time_temp += (Comm_end.tv_sec - Comm_start.tv_sec) + (Comm_end.tv_nsec-Comm_start.tv_nsec)/BILLION;

					clock_gettime(CLOCK_MONOTONIC, &CPU_start);
	                for(i=0; i+1<list_size-1; i+=2) {
	                    if(list[i] > list[i+1]) { swap(&list[i], &list[i+1]); swap_time = 0; }
	                }
					clock_gettime(CLOCK_MONOTONIC, &CPU_end);
                    CPU_time_temp += (CPU_end.tv_sec - CPU_start.tv_sec) + (CPU_end.tv_nsec-CPU_start.tv_nsec)/BILLION;

					clock_gettime(CLOCK_MONOTONIC, &Comm_start);
	                MPI_Recv(&exchange_tail, 1, MPI_FLOAT, rank+1, rank+1, valid_comm, &st);
					clock_gettime(CLOCK_MONOTONIC, &Comm_end);
                    Comm_time_temp += (Comm_end.tv_sec - Comm_start.tv_sec) + (Comm_end.tv_nsec-Comm_start.tv_nsec)/BILLION;
		
	                if(list[list_size-1] > exchange_tail) { list[list_size-1] = exchange_tail; swap_time = 0; }
	            }
				if(rank == proc_size-1) {
					clock_gettime(CLOCK_MONOTONIC, &CPU_start);
					for(i=0; i+1<list_size-1; i+=2) {
	                    if(list[i] > list[i+1]) { swap(&list[i], &list[i+1]); swap_time = 0; }
	                }
					clock_gettime(CLOCK_MONOTONIC, &CPU_end);
                    CPU_time_temp += (CPU_end.tv_sec - CPU_start.tv_sec) + (CPU_end.tv_nsec-CPU_start.tv_nsec)/BILLION;
	            }
				break;
			case 1:
				clock_gettime(CLOCK_MONOTONIC, &CPU_start);
				for(i=0; i+1<list_size; i+=2) {
	                    if(list[i] > list[i+1]) { swap(&list[i], &list[i+1]); swap_time = 0; }
	            }
				clock_gettime(CLOCK_MONOTONIC, &CPU_end);
                CPU_time_temp += (CPU_end.tv_sec - CPU_start.tv_sec) + (CPU_end.tv_nsec-CPU_start.tv_nsec)/BILLION;
	            break;
	        case 2:
	            if(rank == 0) {  
					if(proc_size > 1) {
						clock_gettime(CLOCK_MONOTONIC, &Comm_start);	
						MPI_Isend(&list[list_size-1], 1, MPI_FLOAT, rank+1, rank, valid_comm, &req);
						clock_gettime(CLOCK_MONOTONIC, &Comm_end);
                   		Comm_time_temp += (Comm_end.tv_sec - Comm_start.tv_sec) + (Comm_end.tv_nsec-Comm_start.tv_nsec)/BILLION;
					}	    

					clock_gettime(CLOCK_MONOTONIC, &CPU_start);	            
					for(i=1; i+1<list_size-1; i+=2) {
	                    if(list[i] > list[i+1]) { swap(&list[i], &list[i+1]); swap_time = 0; }
	                }
					clock_gettime(CLOCK_MONOTONIC, &CPU_end);
					CPU_time_temp += (CPU_end.tv_sec - CPU_start.tv_sec) + (CPU_end.tv_nsec-CPU_start.tv_nsec)/BILLION;
		
					if(proc_size > 1) {
						clock_gettime(CLOCK_MONOTONIC, &Comm_start);
	                	MPI_Recv(&exchange_tail, 1, MPI_FLOAT, rank+1, rank+1, valid_comm, &st);
						clock_gettime(CLOCK_MONOTONIC, &Comm_end);
                        Comm_time_temp += (Comm_end.tv_sec - Comm_start.tv_sec) + (Comm_end.tv_nsec-Comm_start.tv_nsec)/BILLION;

	                	if(list[list_size-1] > exchange_tail) { list[list_size-1] = exchange_tail; swap_time = 0; }
	            	}
				}
	            else if(rank < proc_size -1) {  
					clock_gettime(CLOCK_MONOTONIC, &Comm_start);
	                MPI_Isend(&list[list_size-1], 1, MPI_FLOAT, rank+1, rank, valid_comm, &req);
	                MPI_Isend(&list[0], 1, MPI_FLOAT, rank-1, rank, valid_comm, &req);
					clock_gettime(CLOCK_MONOTONIC, &Comm_end);
                    Comm_time_temp += (Comm_end.tv_sec - Comm_start.tv_sec) + (Comm_end.tv_nsec-Comm_start.tv_nsec)/BILLION;	

					clock_gettime(CLOCK_MONOTONIC, &CPU_start);
	                for(i=1; i+1<list_size-1; i+=2) {
	                    if(list[i] > list[i+1]) { swap(&list[i], &list[i+1]); swap_time = 0; }
	                }
					clock_gettime(CLOCK_MONOTONIC, &CPU_end);
                    CPU_time_temp += (CPU_end.tv_sec - CPU_start.tv_sec) + (CPU_end.tv_nsec-CPU_start.tv_nsec)/BILLION;

					clock_gettime(CLOCK_MONOTONIC, &Comm_start);
	                MPI_Recv(&exchange_tail, 1, MPI_FLOAT, rank+1, rank+1, valid_comm, &st);
	                MPI_Recv(&exchange_head, 1, MPI_FLOAT, rank-1, rank-1, valid_comm, &st);
					clock_gettime(CLOCK_MONOTONIC, &Comm_end);
                    Comm_time_temp += (Comm_end.tv_sec - Comm_start.tv_sec) + (Comm_end.tv_nsec-Comm_start.tv_nsec)/BILLION;
					
					if(list[list_size-1] > exchange_tail) { list[list_size-1] = exchange_tail; swap_time = 0; }
	                if(list[0] < exchange_head) { list[0] = exchange_head; swap_time = 0; }
	            }
				else if(rank == proc_size -1) {
					clock_gettime(CLOCK_MONOTONIC, &Comm_start);
					MPI_Isend(&list[0], 1, MPI_FLOAT, rank-1, rank, valid_comm, &req);
					clock_gettime(CLOCK_MONOTONIC, &Comm_end);
                    Comm_time_temp += (Comm_end.tv_sec - Comm_start.tv_sec) + (Comm_end.tv_nsec-Comm_start.tv_nsec)/BILLION;

					clock_gettime(CLOCK_MONOTONIC, &CPU_start);
	                for(i=1; i+1<list_size-1; i+=2) {
	                    if(list[i] > list[i+1]) { swap(&list[i], &list[i+1]); swap_time = 0; }
	                }
					clock_gettime(CLOCK_MONOTONIC, &CPU_end);
                    CPU_time_temp += (CPU_end.tv_sec - CPU_start.tv_sec) + (CPU_end.tv_nsec-CPU_start.tv_nsec)/BILLION;

					clock_gettime(CLOCK_MONOTONIC, &Comm_start);
	                MPI_Recv(&exchange_head, 1, MPI_FLOAT, rank-1, rank-1, valid_comm, &st);
					clock_gettime(CLOCK_MONOTONIC, &Comm_end);
                    Comm_time_temp += (Comm_end.tv_sec - Comm_start.tv_sec) + (Comm_end.tv_nsec-Comm_start.tv_nsec)/BILLION;
	
	                if(list[0] < exchange_head) { list[0] = exchange_head; swap_time = 0; }	
				}
	            break;
	        case 3:	//No worry about proc_size = 1, since rank==0 do local sort only
				if(rank == 0) {
					clock_gettime(CLOCK_MONOTONIC, &CPU_start);
					for(i=1; i+1<list_size; i+=2) {
	                    if(list[i] > list[i+1]) { swap(&list[i], &list[i+1]); swap_time = 0; }
	                }	
					clock_gettime(CLOCK_MONOTONIC, &CPU_end);
                    CPU_time_temp += (CPU_end.tv_sec - CPU_start.tv_sec) + (CPU_end.tv_nsec-CPU_start.tv_nsec)/BILLION;
				}
	            if(rank > 0) {
					clock_gettime(CLOCK_MONOTONIC, &Comm_start);
					MPI_Isend(&list[0], 1, MPI_FLOAT, rank-1, rank, valid_comm, &req);
					clock_gettime(CLOCK_MONOTONIC, &Comm_end);
                    Comm_time_temp += (Comm_end.tv_sec - Comm_start.tv_sec) + (Comm_end.tv_nsec-Comm_start.tv_nsec)/BILLION;

					clock_gettime(CLOCK_MONOTONIC, &CPU_start);
	                for(i=1; i+1<list_size; i+=2) {
	                    if(list[i] > list[i+1]) { swap(&list[i], &list[i+1]); swap_time = 0; }
	                }
					clock_gettime(CLOCK_MONOTONIC, &CPU_end);
                    CPU_time_temp += (CPU_end.tv_sec - CPU_start.tv_sec) + (CPU_end.tv_nsec-CPU_start.tv_nsec)/BILLION;
	
					clock_gettime(CLOCK_MONOTONIC, &Comm_start);
					MPI_Recv(&exchange_head, 1, MPI_FLOAT, rank-1, rank-1, valid_comm, &st);
					clock_gettime(CLOCK_MONOTONIC, &Comm_end);
                    Comm_time_temp += (Comm_end.tv_sec - Comm_start.tv_sec) + (Comm_end.tv_nsec-Comm_start.tv_nsec)/BILLION;

	                if(list[0] < exchange_head) { list[0] = exchange_head; swap_time = 0; }
	            }
	    }
		MPI_Barrier(valid_comm);	
		
		int tmp = swap_time;
		clock_gettime(CLOCK_MONOTONIC, &Comm_start);
		MPI_Allreduce(&tmp, &swap_time, 1, MPI_INT, MPI_BAND, valid_comm);
		clock_gettime(CLOCK_MONOTONIC, &Comm_end);
        if(proc_size > 1) Comm_time_temp += (Comm_end.tv_sec - Comm_start.tv_sec) + (Comm_end.tv_nsec-Comm_start.tv_nsec)/BILLION;
		//printf("Allreduce time: %lf\n", Comm_time_temp); 
	}

	/*
	printf("Rank: %d, ", rank);
	for(i=0; i<list_size; i++) printf("%f ", list[i]);
	printf("\n");
	
	while(swap) {
		swap = 0;
		
		//Even Phase
		if(tail_gindex%2 == 0) {
			if(rank+1 < proc_size) {
				MPI_Isend(&list[list_size-1], 1, MPI_FLOAT, rank+1, rank, MPI_COMM_WORLD);
				MPI_Recv(&exchange, 1, MPI_FLOAT, rank+1, rank+1, MPI_COMM_WORLD);
	*/
	clock_gettime(CLOCK_MONOTONIC, &IO_start);
	rc = MPI_File_open(valid_comm, argv[3], MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);	
	MPI_File_write_at(fh, sizeof(float)*head_gindex, list, list_size, MPI_FLOAT, &st);
	MPI_File_close(&fh);
	clock_gettime(CLOCK_MONOTONIC, &IO_end);
    IO_time_temp += (IO_end.tv_sec - IO_start.tv_sec) + (IO_end.tv_nsec-IO_start.tv_nsec)/BILLION;
	//printf("Rank: %d, IO_write: %lf\n", rank, IO_time_temp);

	MPI_Barrier(valid_comm);
	MPI_Reduce(&IO_time_temp, &IO_time, 1, MPI_DOUBLE, MPI_MAX, 0, valid_comm);
	MPI_Reduce(&CPU_time_temp, &CPU_time, 1, MPI_DOUBLE, MPI_MAX, 0, valid_comm);
    MPI_Reduce(&Comm_time_temp, &Comm_time, 1, MPI_DOUBLE, MPI_MAX, 0, valid_comm);
	if(rank == 0) {
        printf("IO_time: %lf\nCPU_time: %lf\nComm_time: %lf\n", IO_time, CPU_time, Comm_time);
    }
	//printf("Rank: %d, %lf, %lf\n", rank, IO_time_temp, IO_time); 

	free(list);
	MPI_Comm_free(&valid_comm);
	MPI_Finalize();
	return 0;
}
