namespace NboxTrainer.Services
{
    public class ServiceStatePredictor : IServiceState<StatePrediction>
    {
        public ServiceStatePredictor()
        {
            _locker = new object();
            _currentState = StatePrediction.None;
        }

        private StatePrediction _currentState = StatePrediction.None;
        private object _locker;

        public delegate void OnStateChanged(StatePrediction state);
        public event OnStateChanged onStateChanged;

        public void setState(StatePrediction newState)
        {
            lock (_locker) { _currentState = newState; onStateChanged?.Invoke(newState); }
        }

        public StatePrediction getState()
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
                _currentState = StatePrediction.None;
                onStateChanged?.Invoke(_currentState);
            }
        }
    }
}