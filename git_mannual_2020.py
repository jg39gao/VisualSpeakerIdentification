
**** INSTALL git ********************************************************

----- MAC ------
# STEP1-- Install homebrew (a package management sys, maybe just like anaconda). 

/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
brew doctor

# STEP2 install git 

brew install git

----- Window -----

# Download Git from Git for Windows (https://gitforwindows.org/) and install it.

----- Linux -----
# Debian-based linux systems
sudo apt update
sudo apt upgrade
sudo apt install git

# Red Hat-based linux systems
sudo yum upgrade
sudo yum install git




****** git STARTER ******************************************************


# Choose a local directory to Clone a remote repository , when be asked psw, type your computer psw
git clone git@github.com:jg39gao/VisualSpeakerIdentification.git
'''after this step, the repository has been cloned into your new directory '''



****** git MainCourse ******************************************************

# Pull from the remote repo. (Technically fetch + merge remote changes first.)

git pull 

# Add the change to stage area 
git add * 

# Let commit the change to the HEAD . HEAD is still in your local repo , it needs to be pushed to the remote one.
git commit -m 'your commit messages'

# Push the commitment to remote origin. 
git push <origin master> 


	
	****** Branch  ******************************************************
	# Check current branches
	git branch 
	# Create a new branch 'dev' and switch to it.  The branch is isolated from others until you push it to the remote .
	git checkout -b dev 

	# Switch back to master 
	git checkout master 

	# del a branch 
	git branch -d dev 

	# Push branch to remote repo. **NOT RECOMMENDED**
	'''every time you just need to merge your own branch to master , then push master to the remote repo '''
	git push origin <branch> 


	****** Merge  ******************************************************

	# Merge branch dev to your current branch , say, master 
	git merge dev 
	# conflict between two branches leads to Automatic merge failing; fix conflicts and then commit the result (add and commit ).

	# You also can Check the difference. 
	git diff <currentBranch> <merge branch>



****** ABOUT SSH  ******************************************************
# --- check if you have got a rsa key (id_rsa, id_rsa.pub)，
cat ~/.ssh/id_rsa.pub

# //copy to clipboard for mac. Add this SSH key in your github account/ setting 
pbcopy < ~/.ssh/id_rsa.pub
# If you don't have it,  then Create one :
ssh-keygen -t rsa -C "example@gmail.com"

# ****TRIFLE** : when encountering everytime passphrase requirement. do this:
# This will ask you for the passphrase, enter it and it won't ask again.
ssh-add ~/.ssh/id_rsa &>/dev/null

# when git doesn't work and prompt:
>> fatal: bad config line 1 in file /Users/Username/.gitconfig
# solution is: 
Delete your ~/.gitconfig and manipulate it using the git config command at the terminal, as explained in the tutorial.

'''
-------------
# @References:
-------------
 https://rogerdudler.github.io/git-guide/ 
