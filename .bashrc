# ~/.bashrc for MAGs-CodeDev OMP Sandbox

# Ensure bun and nvim are in PATH
export PATH="/root/.bun/bin:/opt/nvim-linux64/bin:$PATH"

# Re-link local extensions on every boot (~/.omp is mounted from host,
# which wipes any plugin links from build time)
[ -d /ext ] && omp plugin link /ext 2>/dev/null
[ -d /opt/pi-nvim-bridge ] && omp plugin link /opt/pi-nvim-bridge 2>/dev/null

# Re-install npm-scoped plugins if the host wiped them
for pkg in \
    pi-lens \
    pi-context-prune \
    @code-yeongyu/pi-rules \
    @narumitw/pi-statusline \
    @aliou/pi-guardrails \
    @gotgenes/pi-permission-system \
    @isac322/pi-codegraph \
; do
    omp plugin list 2>/dev/null | grep -q "$(basename $pkg)" || omp install "$pkg" 2>/dev/null
done

# Use Neovim as the external editor for OMP
export EDITOR=nvim
export VISUAL=nvim
export PI_NVIM_APPNAME=pi-bridge

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
