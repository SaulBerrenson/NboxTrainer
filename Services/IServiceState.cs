namespace NboxTrainer.Services
{
    public interface IServiceState<TEnum>
    {
        public void setState(TEnum newState);

        public TEnum getState();

        public void resetState();

    }
}