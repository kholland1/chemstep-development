# chemstep


**GitHub instructions**
1. Ideas for changes are listed in Issues. Individually or as a team choose an Issue to tackle, and assign it to yourself. Create your own branch of the code based off of the main.

<img width="327" height="140" alt="Screenshot 2025-10-17 at 12 07 26 PM" src="https://github.com/user-attachments/assets/c5bab9a4-bc10-44ff-81bf-f5f194751109" />



2. Make edits to the code as needed. Most changes will likely take place in `algo.py` or `search_job.py`. Ideally your changes should be linked to a parameter passed to the main CSAlgo object so that we can try multiple different values. The default behavior if your new parameter is not passed should always be to run the algorithm as it currently exists.

  
3. Track major changes you make in the comments section of the Issue.


**Running ChemSTEP on Wynton**

1. Make a directory under your name or team within `/wynton/group/bks/work/chemstep_dev`



     <img width="303" height="33" alt="Screenshot 2025-10-17 at 3 46 10 PM" src="https://github.com/user-attachments/assets/e64df8cb-34b9-48fd-9aa9-319106b39354" />


     `cd <your_directory>`

    
2. Make a directory for any major change applied (Issue):
   

    `mkdir <name_of_change_applied>`
   

   
    <img width="352" height="70" alt="Screenshot 2025-10-17 at 3 47 44 PM" src="https://github.com/user-attachments/assets/1dc23f10-dbeb-465b-8d13-2a177408559e" />


     `cd <name_of_change_applied>`

3. Pull github branch

    `git clone -b <branch-name> --single-branch https://github.com/bwhall61/chemstep-development.git`

4. Make code changes to your branch. Make sure to commit often when you make changes!

   `git add .`
   
   `git commit -m "Added minTD threshold to SearchJob"`


5. Create a python environment for your modified version of ChemSTEP in your base directory:
   
    `python3 -m venv <name_of_new_environment>` 


6. Activate your new envionment:
   
    `source <name_of_new_environment>/bin/activate`

7. Install your modified version of ChemSTEP. In your chemstep branch run:

    `pip install .`


8. Make a working directory on the same level as the venv:

    `mkdir chemstep_run`
    

   
    <img width="385" height="87" alt="Screenshot 2025-10-17 at 3 48 09 PM" src="https://github.com/user-attachments/assets/be97abf2-e8e3-4007-a57a-a3e35ea49ac8" />


    `cd chemstep_run`


9. Copy in necessary files to the working chemstep directory: 

    `cp /wynton/group/bks/work/chemstep_dev/scripts/params.txt .`

    `cp /wynton/group/bks/work/chemstep_dev/scripts/run_chemstep.py .`

    `cp /wynton/group/bks/work/chemstep_dev/scripts/launch_chemstep_as_job.sh .`
   

10. Update `launch_chemstep_as_job.sh` and `run_chemstep.py`

   In `launch_chemstep_as_job.sh` update <path-to-your-enviornment> with the path to your environment.
   
   In `run_chemstep.py` update <path-to-your-environment> with the path to your environment. Also add any additional parameters needed by your changes to the CSAlgo object.

**some things that should NOT change:**

      dockfiles_path=/wynton/group/bks/work/bwhall61/mor_chemstep/DOCK/dockfiles
      
      docking_method="auto"
      
      scheduler='sge'
      
      smi_id_prefix='MOL'
      
      score_db="/wynton/group/bks/work/bwhall61/CHEMSTEP_MEGA_FIXING_FOLDER/no_strain_mor_scores.db"

      ignore_seeds=True

      track_beacon_orig=True
      
11. Submit job to scheduler:

    `qsub launch_chemstep_as_job.sh`

12. After submission, also `git push` your changes to github and let Brendan know so he can look over your code while it's running.

13. Wait for the algorithm to run for 15-20 rounds. You can look in your chemstep folder to see how many rounds have been run. After you have many rounds, delete your chemstep job to stop it from running (see SGE commands below) 



**helpful SGE commands:**

        qstat - see jobs running and queued 
        
        qstat -j - see more info on jobs running
        
        qdel - delete jobs 

**Plotting**

1. First copy the following files into your chemstep folder
   `cp /wynton/group/bks/work/bwhall61/CHEMSTEP_MEGA_FIXING_FOLDER/mor_13M_indices_round_0.npy <your-chemstep-run-folder>/indices_round_0.npy`
   
   `cp /wynton/group/bks/work/bwhall61/CHEMSTEP_MEGA_FIXING_FOLDER/scores_round_0.txt <your-chemstep-run-folder>`
   
2. Make a plotting directory in your chemstep folder and copy the plotting script into it

   `mkdir <your-chemstep-run-folder>/plotting`
   
   
   `cp /wynton/group/bks/work/chemstep_dev/scripts/chemstep_plots_array.sh <your-chemstep-run-folder>/plotting`

3. Move into the plotting folder and launch the plotting job

   
   `cd <your-chemstep-run-folder>/plotting`
   
   `qsub chemstep_plots_array.sh`
  

4. Upload plots back into GitHub Issue comment section!


**notes**
1. every Issue/major change applied needs to be run in its own seperate directory for plotting purposes
   

2. do not make changes to `autodock_algo.py` or change any parameters related to building or docking
   

3. in general, jobs will run as follows 


   Main ChemSTEP job (will run constantly): reads input files (DOCK scores and line-matched indices NumPy arrays) and assigns beacons. after chaining, will gather SMILES for prioritization 



   ChainingLog sub-jobs: once beacons are assigned, ChemSTEP will launch a series of sub-jobs that calculate the Tanimoto distances of every molecule remaining in the library to the assigned beacons. Calculated distances to the NEAREST beacon (minimum-minimum Td) are updated in the mintds_*.npy files within the /output directory


   

   Round_N building: generates 3D conformations, precompute solvation, etc of prioritized molecules 
   

   Round_N docking: docks prioritized molecules to the receptor. Scores from docking are given back to ChemSTEP to iniate the next round


