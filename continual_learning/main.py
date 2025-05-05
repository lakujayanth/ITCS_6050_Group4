import logging
from plots import plot_transfer_comparison, plot_meta_adaptation, plot_ewc_forgetting

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

if __name__ == '__main__':
    logging.info("=== Experiments Start ===")
    plot_transfer_comparison()
    plot_meta_adaptation()
    plot_ewc_forgetting()
    logging.info("=== Experiments Complete ===")