namespace NboxTrainer.Services
{
    public class ServiceStateML : IServiceState<StateML>
    {
        public ServiceStateML()
        {
            _locker = new object();
            _currentState = StateML.None;
        }

        private StateML _currentState = StateML.None;
        private object _locker;

        public delegate void OnStateChanged(StateML state);
        public event OnStateChanged onStateChanged;

        public void setState(StateML newState = StateML.None)
        {
            lock (_locker) { _currentState = newState; onStateChanged?.Invoke(newState); }
        }
        public StateML getState()
        {
            lock (_locker)
            {
                return _currentState;
            }
        }

        public void resetState()
        {
            lock (_locker)
            {
                _currentState = StateML.None;
                onStateChanged?.Invoke(_currentState);
            }
        }

    }
}