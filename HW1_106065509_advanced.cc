#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
#include <algorithm>
#define BILLION 1E9

float* list;
float* a;
float* b;
void merge(int s,int m,int e){
    int i,j;
    for(i=s;i<=m;++i)a[i] = list[i];
    for(j=m+1;j<=e;++j)b[j] = list[j];
    for(i=s,j=m+1 ; i<=m && j<=e ;){
        if(a[i] <= b[j])list[s++] = a[i++];
        else list[s++] = b[j++];
    }
    while(i<=m) list[s++] = a[i++];
    while(j<=e) list[s++] = b[j++];
}

void mergeSort(int s,int e){
    if(s>=e)return ;
    int m = (s+e)/2;
    mergeSort(s,m);
    mergeSort(m+1,e);
    merge(s,m,e);
}



int main(int argc, char *argv[])
{
	int rank, proc_size, num_size, list_size, head_gindex, rc;
	MPI_File fh;
	MPI_Status st;
	MPI_Comm valid_comm, other_comm;
	num_size = atoi(argv[1]);
	struct timespec IO_start, IO_end, CPU_start, CPU_end, Comm_start, Comm_end;
    double IO_time, IO_time_temp, CPU_time, CPU_time_temp, Comm_time, Comm_time_temp;
    IO_time = IO_time_temp = CPU_time = CPU_time_temp = Comm_time = Comm_time_temp = 0;
	
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &proc_size);
	if(proc_size > num_size) proc_size = num_size;
	if(rank >= proc_size) {
		MPI_Comm_split(MPI_COMM_WORLD, rank / proc_size, rank, &other_comm);
        MPI_Comm_free(&other_comm);
        MPI_Finalize();
        return 0;
	}
	else MPI_Comm_split(MPI_COMM_WORLD, rank / proc_size, rank, &valid_comm);

	if(rank < (num_size % proc_size)) {
		list_size = num_size / proc_size + 1;
		head_gindex = list_size * rank;
	}
	else {
		list_size = num_size / proc_size;
		head_gindex = (list_size+1)*(num_size%proc_size) + list_size*(rank-(num_size%proc_size));
	}
	list = (float*)malloc(sizeof(float)*list_size);
	a = (float*)malloc(sizeof(float)*list_size);
	b = (float*)malloc(sizeof(float)*list_size);
	
	clock_gettime(CLOCK_MONOTONIC, &IO_start);
	rc = MPI_File_open(valid_comm, argv[2], MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
	MPI_File_read_at(fh, sizeof(float)*head_gindex, list, list_size, MPI_FLOAT, &st);
	//printf("Rank: %d, list[0]: %f\n", rank, list[0]);
	MPI_File_close(&fh);
	clock_gettime(CLOCK_MONOTONIC, &IO_end);
    IO_time_temp += (IO_end.tv_sec - IO_start.tv_sec) + (IO_end.tv_nsec-IO_start.tv_nsec)/BILLION;
	
	clock_gettime(CLOCK_MONOTONIC, &CPU_start);
	//mergeSort(0, list_size-1);	
	std::sort(list, list+list_size);
	clock_gettime(CLOCK_MONOTONIC, &CPU_end);
    CPU_time_temp += (CPU_end.tv_sec - CPU_start.tv_sec) + (CPU_end.tv_nsec-CPU_start.tv_nsec)/BILLION;

	int swap_time, pre_list_size, i, j, k;
	float* exchange_head;
	float* tmp_list;
	MPI_Request req;
	if(rank > 0) {
		pre_list_size = (rank-1 < (num_size%proc_size))? num_size / proc_size + 1 : num_size / proc_size;
		exchange_head = (float*)malloc(sizeof(float)*pre_list_size);
		tmp_list = (float*)malloc(sizeof(float)*(pre_list_size+list_size));
	}
	swap_time = (proc_size <= 1)? 0 : 1;	//only one process, then mergesort is enough
	//MPI_Barrier(MPI_COMM_WORLD);

	while(swap_time) {
		swap_time = 0;
		if(rank%2 == 0 && rank < proc_size-1) {
			clock_gettime(CLOCK_MONOTONIC, &Comm_start);
			MPI_Isend(list, list_size, MPI_FLOAT, rank+1, rank, valid_comm, &req);
			clock_gettime(CLOCK_MONOTONIC, &Comm_end);
            Comm_time_temp += (Comm_end.tv_sec - Comm_start.tv_sec) + (Comm_end.tv_nsec-Comm_start.tv_nsec)/BILLION;

			clock_gettime(CLOCK_MONOTONIC, &Comm_start);
			MPI_Recv(list, list_size, MPI_FLOAT, rank+1, rank+1, valid_comm, &st);
			clock_gettime(CLOCK_MONOTONIC, &Comm_end);
			Comm_time_temp += (Comm_end.tv_sec - Comm_start.tv_sec) + (Comm_end.tv_nsec-Comm_start.tv_nsec)/BILLION;
		}
		else if(rank%2 == 1) {
			clock_gettime(CLOCK_MONOTONIC, &Comm_start);
			MPI_Recv(exchange_head, pre_list_size, MPI_FLOAT, rank-1, rank-1, valid_comm, &st);
			clock_gettime(CLOCK_MONOTONIC, &Comm_end);
            Comm_time_temp += (Comm_end.tv_sec - Comm_start.tv_sec) + (Comm_end.tv_nsec-Comm_start.tv_nsec)/BILLION;
			//std::copy(exchange_head, exchange_head+pre_list_size, tmp_list);
			//std::copy(list, list+list_size, tmp_list+pre_list_size);
			i = j = k = 0;
			clock_gettime(CLOCK_MONOTONIC, &CPU_start);
			while(i<pre_list_size && j<list_size) {
				if(exchange_head[i] <= list[j]) tmp_list[k++] = exchange_head[i++];
				else { tmp_list[k++] = list[j++]; swap_time = 1;}
			}
			while(i < pre_list_size) tmp_list[k++] = exchange_head[i++];
			while(j < list_size) tmp_list[k++] = list[j++];
			clock_gettime(CLOCK_MONOTONIC, &CPU_end);
		    CPU_time_temp += (CPU_end.tv_sec - CPU_start.tv_sec) + (CPU_end.tv_nsec-CPU_start.tv_nsec)/BILLION;
			
			clock_gettime(CLOCK_MONOTONIC, &Comm_start);			
			MPI_Isend(tmp_list, pre_list_size, MPI_FLOAT, rank-1, rank, valid_comm, &req);
			clock_gettime(CLOCK_MONOTONIC, &Comm_end);
            Comm_time_temp += (Comm_end.tv_sec - Comm_start.tv_sec) + (Comm_end.tv_nsec-Comm_start.tv_nsec)/BILLION;

			//memcpy(list, tmp_list+pre_list_size, sizeof(float)*list_size);	
			std::copy(tmp_list+pre_list_size, tmp_list+(pre_list_size+list_size), list);
		}
		MPI_Barrier(valid_comm);
		/*	
		printf("Rank: %d, Even: ", rank);
    	for(i=0; i<list_size; i++) printf("%f ", list[i]);
    	printf("\n");
		*/
		if(rank%2 == 1 && rank < proc_size -1) {
			clock_gettime(CLOCK_MONOTONIC, &Comm_start);
			MPI_Isend(list, list_size, MPI_FLOAT, rank+1, rank, valid_comm, &req);
			clock_gettime(CLOCK_MONOTONIC, &Comm_end);
            Comm_time_temp += (Comm_end.tv_sec - Comm_start.tv_sec) + (Comm_end.tv_nsec-Comm_start.tv_nsec)/BILLION;	

			clock_gettime(CLOCK_MONOTONIC, &Comm_start);
			MPI_Recv(list, list_size, MPI_FLOAT, rank+1, rank+1, valid_comm, &st);
			clock_gettime(CLOCK_MONOTONIC, &Comm_end);
            Comm_time_temp += (Comm_end.tv_sec - Comm_start.tv_sec) + (Comm_end.tv_nsec-Comm_start.tv_nsec)/BILLION;
		}
		else if(rank%2 == 0 && rank > 0) {
			clock_gettime(CLOCK_MONOTONIC, &Comm_start);
			MPI_Recv(exchange_head, pre_list_size, MPI_FLOAT, rank-1, rank-1, valid_comm, &st);
			clock_gettime(CLOCK_MONOTONIC, &Comm_end);
            Comm_time_temp += (Comm_end.tv_sec - Comm_start.tv_sec) + (Comm_end.tv_nsec-Comm_start.tv_nsec)/BILLION;			

			i = j = k = 0;
            clock_gettime(CLOCK_MONOTONIC, &CPU_start);
			while(i<pre_list_size && j<list_size) {
                if(exchange_head[i] <= list[j]) tmp_list[k++] = exchange_head[i++];
                else { tmp_list[k++] = list[j++]; swap_time = 1;}
            }
            while(i < pre_list_size) tmp_list[k++] = exchange_head[i++];
            while(j < list_size) tmp_list[k++] = list[j++];
			clock_gettime(CLOCK_MONOTONIC, &CPU_end);
            CPU_time_temp += (CPU_end.tv_sec - CPU_start.tv_sec) + (CPU_end.tv_nsec-CPU_start.tv_nsec)/BILLION;
	
			clock_gettime(CLOCK_MONOTONIC, &Comm_start);
            MPI_Isend(tmp_list, pre_list_size, MPI_FLOAT, rank-1, rank, valid_comm, &req);
            clock_gettime(CLOCK_MONOTONIC, &Comm_end);
            Comm_time_temp += (Comm_end.tv_sec - Comm_start.tv_sec) + (Comm_end.tv_nsec-Comm_start.tv_nsec)/BILLION;
			
			//memcpy(list, tmp_list+pre_list_size, sizeof(float)*list_size);
			std::copy(tmp_list+pre_list_size, tmp_list+(pre_list_size+list_size), list);
		}
		MPI_Barrier(valid_comm);
		
		int tmp = swap_time;
		clock_gettime(CLOCK_MONOTONIC, &Comm_start);
		MPI_Allreduce(&tmp, &swap_time, 1, MPI_INT, MPI_BOR, valid_comm);
		clock_gettime(CLOCK_MONOTONIC, &Comm_end);
        Comm_time_temp += (Comm_end.tv_sec - Comm_start.tv_sec) + (Comm_end.tv_nsec-Comm_start.tv_nsec)/BILLION;
		//printf("Allreduce: %lf\n", Comm_time_temp);
	}
	/*
	printf("Rank: %d, Odd: ", rank);
 	for(i=0; i<list_size; i++) printf("%f ", list[i]);
  	printf("\n");
	*/ 
	//MPI_Barrier(MPI_COMM_WORLD);
	
	clock_gettime(CLOCK_MONOTONIC, &IO_start);	
	rc = MPI_File_open(valid_comm, argv[3], MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
    MPI_File_write_at(fh, sizeof(float)*head_gindex, list, list_size, MPI_FLOAT, &st);
    MPI_File_close(&fh);
	clock_gettime(CLOCK_MONOTONIC, &IO_end);
    IO_time_temp += (IO_end.tv_sec - IO_start.tv_sec) + (IO_end.tv_nsec-IO_start.tv_nsec)/BILLION;	

	MPI_Barrier(valid_comm);
	MPI_Reduce(&IO_time_temp, &IO_time, 1, MPI_DOUBLE, MPI_MAX, 0, valid_comm);
	MPI_Reduce(&CPU_time_temp, &CPU_time, 1, MPI_DOUBLE, MPI_MAX, 0, valid_comm);
	MPI_Reduce(&Comm_time_temp, &Comm_time, 1, MPI_DOUBLE, MPI_MAX, 0, valid_comm);
	if(rank == 0) {
		printf("IO_time: %lf\nCPU_time: %lf\nComm_time: %lf\n", IO_time, CPU_time, Comm_time);
	}
	free(list);
	MPI_Comm_free(&valid_comm);
	MPI_Finalize();
	return 0;
}
