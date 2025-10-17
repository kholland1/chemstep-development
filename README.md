# chemstep


**GitHub instructions**
1. Ideas for changes are listed in Issues. Individually or as a team choose an Issue to tackle, and assign it to yourself. Create your own branch of the code based off of the main.

<img width="327" height="140" alt="Screenshot 2025-10-17 at 12 07 26 PM" src="https://github.com/user-attachments/assets/c5bab9a4-bc10-44ff-81bf-f5f194751109" />



2. Make edits as needed. Most changes will likely take place in `algo.py` or `search_job.py`.

  
3. Track changes you make in the comments section of the Issue. 


**Running ChemSTEP on Wynton**

1. Make a directory under your name or team within `/wynton/group/bks/work/chemstep_dev`



     <img width="303" height="33" alt="Screenshot 2025-10-17 at 3 46 10 PM" src="https://github.com/user-attachments/assets/e64df8cb-34b9-48fd-9aa9-319106b39354" />


     `cd your_directory`
    
2. Make a directory for any major change applied (Issue):

    `mkdir name_of_change_applied`
   

   
    <img width="352" height="70" alt="Screenshot 2025-10-17 at 3 47 44 PM" src="https://github.com/user-attachments/assets/1dc23f10-dbeb-465b-8d13-2a177408559e" />


     `cd name_of_change_applied`


3. Create a virtual python environment for your ChemSTEP version in your base directory:
   
    `python -m venv name_of_new_environment` 


4. Source new virtual envionment:
   
    `source /wynton/group/bks/work/chemstep_dev/your/venv/bin/activate`



5. Pull code branch with changes:
   
    ` ask brendan what this command is`



6. Make a working directory on the same level as the venv:

    `mkdir chemstep`
    

   
    <img width="385" height="87" alt="Screenshot 2025-10-17 at 3 48 09 PM" src="https://github.com/user-attachments/assets/be97abf2-e8e3-4007-a57a-a3e35ea49ac8" />


    `cd chemstep`



7. Copy in necessary files to the working chemstep directory: 

    `cp /wynton/group/bks/work/chemstep_dev/scripts/params.txt .`

    `cp /wynton/group/bks/work/chemstep_dev/scripts/run_chemstep.py .`

    `cp /wynton/group/bks/work/chemstep_dev/scripts/launch_chemstep_as_job.sh .`
   


8. Make any edits necessary within `run_chemstep.py`. ** if changes were made to `search_jobs.py` you need to update the `python_exec=/path`
    

**some things that should NOT change:**

      dockfiles_path=/wynton/group/bks/work/bwhall61/mor_chemstep/DOCK/dockfiles
      
      docking_method="auto"
      
      scheduler='sge'
      
      smi_id_prefix='MOL'
      
      score_db="/wynton/group/bks/work/bwhall61/CHEMSTEP_MEGA_FIXING_FOLDER/no_strain_mor_scores.db"

      ignore_seeds=True

      track_beacon_orig=True
      

9. Edit `launch_chemstep_as_job.sh` to call the virtual environment created in step 3.

10. Submit job to scheduler:

    `qsub launch_chemstep_as_job.sh` 

**helpful SGE commands:**

        qstat - see jobs running and queued 
        
        qstat -j - see more info on jobs running
        
        qdel - delete jobs 

**Plotting**

1. 

2. Upload plots back into GitHub Issue comment section.


**notes**
1. every Issue/major change applied needs to be run in its own seperate directory for plotting purposes
   

2. do not make changes to `autodock_algo.py` or change any parameters related to building or docking
   

3. in general, jobs will run as follows 


   Main ChemSTEP job (will run constantly): reads input files (DOCK scores and line-matched indices NumPy arrays) and assigns beacons. after chaining, will gather SMILES for prioritization 



   ChainingLog sub-jobs: once beacons are assigned, ChemSTEP will launch a series of sub-jobs that calculate the Tanimoto distances of every molecule remaining in the library to the       assigned beacons. Calculated distances to the NEAREST beacon (minimum-minimum Td) are updated in the mintddistrib_*.npy files within the /output directory


   

   Round_N building: generates 3D conformations, precompute solvation, etc of prioritized molecules 
   

   Round_N docking: docks prioritized molecules to the receptor. Scores from docking are given back to ChemSTEP to iniate the next round


