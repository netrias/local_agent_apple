# Installs oMLX LLM server and pi coding agent, with web search capability.
echo "Installing homebrew package manager..."
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew update && brew upgrade
brew install nvm

echo "Configuring nvm so Node/npm can be upgraded..."
export NVM_DIR="$HOME/.nvm"
mkdir -p "$NVM_DIR"
NVM_BREW_PREFIX="$(brew --prefix nvm)"
[ -s "$NVM_BREW_PREFIX/nvm.sh" ] && \. "$NVM_BREW_PREFIX/nvm.sh"

if ! grep -q 'NVM_DIR' ~/.zshrc 2>/dev/null; then
  {
    echo ''
    echo 'export NVM_DIR="$HOME/.nvm"'
    echo "[ -s \"$NVM_BREW_PREFIX/nvm.sh\" ] && \\. \"$NVM_BREW_PREFIX/nvm.sh\""
    echo "[ -s \"$NVM_BREW_PREFIX/etc/bash_completion.d/nvm\" ] && \\. \"$NVM_BREW_PREFIX/etc/bash_completion.d/nvm\""
  } >> ~/.zshrc
fi

echo "Installing oMLX LLM server..."
curl -sSL -o oMLX-0.6.4-macos26-27.dmg https://github.com/jundot/omlx/releases/download/v0.6.4/oMLX-0.6.4-macos26-27.dmg
VOLUME=$(hdiutil attach oMLX-0.6.4-macos26-27.dmg | tail -n 1 | awk -F'\t' '{print $NF}')
echo "Mounted at: $VOLUME"
echo "Copying the application..."
#cp -Rp "$VOLUME"/*.app /Applications/
echo "Unmounting and cleaning up oMLX image..."
hdiutil detach "$VOLUME"
echo "oMLX installation complete!"

echo "Installing Pi coding agent with web search capability..."
nvm install --lts
nvm use --lts
nvm alias default 'lts/*'
npm install -g npm@latest
echo "Node $(node -v) / npm $(npm -v) active."
#curl -fsSL https://pi.dev/install.sh | sh
pi install npm:pi-smart-web-search
pi install npm:pi-smart-fetch
echo "Stack installation complete!"
. ~/.zshrc
echo "Run terminal command 'pi' to start the Pi coding agent."