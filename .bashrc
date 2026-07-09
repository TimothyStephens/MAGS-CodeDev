# ~/.bashrc for MAGS-CodeDev OMP Sandbox

# Ensure bun is in PATH (required for OMP)
export PATH="/root/.bun/bin:$PATH"

# Re-link extension on every boot (~/.omp is mounted from host,
# which wipes any plugin links from build time)
[ -d /ext ] && omp plugin link /ext 2>/dev/null

alias ll='ls -lAh --color=auto'
alias la='ls -A --color=auto'
alias l='ls -CF --color=auto'
alias gs='git status'
alias gc='git commit'
alias glog='git log --oneline --graph --decorate'

PS1='\[\e[01;32m\]\u@\h\[\e[00m\]:\[\e[01;34m\]\w\[\e[00m\]\$ '

if [ -t 1 ]; then
    export TERM=xterm-256color
fi
