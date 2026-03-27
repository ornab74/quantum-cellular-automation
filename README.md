```

#include <algorithm>
#include <cmath>
#include <complex>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using cd = std::complex<double>;

struct SimConfig {
    int rows = 8, cols = 8, ns = 4, nd = 3, steps = 240;
    double dt = 0.02, measurement_strength = 0.06, future_attractor_bias = 0.035, repair_strength = 0.012;
    unsigned seed = 271828;
};

struct DensityCell {
    int ns, nd;
    std::vector<std::vector<cd>> rho_s, rho_d;
    double topo = 0.0, phase_bias = 0.0;
    DensityCell(int ns_=4, int nd_=3): ns(ns_), nd(nd_),
        rho_s(ns_, std::vector<cd>(ns_, 0.0)),
        rho_d(nd_, std::vector<cd>(nd_, 0.0)) {}
};

struct StepMetrics {
    int step = 0;
    double field_phase=0, field_amplitude=0, field_frequency=0, phase_lock=0, moving_whole_coherence=0;
    double mean_active_info=0, mean_implicate_memory=0, symbol_purity=0, delay_purity=0, symbol_coherence=0;
    double topo_mean=0, topo_std=0, entanglement_proxy=0, observable_z=0, observable_semantic=0;
};

class ChrononFieldV8 {
public:
    explicit ChrononFieldV8(const SimConfig& cfg)
        : cfg_(cfg), rows_(cfg.rows), cols_(cfg.cols), ns_(cfg.ns), nd_(cfg.nd),
          cells_(rows_ * cols_, DensityCell(ns_, nd_)),
          active_info_(rows_ * cols_, 0.0),
          implicate_memory_(rows_ * cols_, 0.0),
          future_pull_(rows_ * cols_, 0.0),
          rng_(cfg.seed), gauss_(0.0, 1.0) {}

    void seed_region(int r0, int r1, int c0, int c1, int a, int b, double mix,
                     double gx, double gy, int delay_mode, double topo_target) {
        for (int r=r0; r<=r1; ++r) for (int c=c0; c<=c1; ++c) {
            auto &cell = at(r,c);
            zero_matrix(cell.rho_s); zero_matrix(cell.rho_d);
            double phi = gx*c + gy*r;
            cell.phase_bias = phi;
            cell.topo = topo_target;

            std::vector<cd> psi_s(ns_, 0.0);
            psi_s[a] = std::polar(std::sqrt(1.0 - mix), phi);
            psi_s[b] = std::polar(std::sqrt(mix), -0.5 * phi);
            pure_to_density(psi_s, cell.rho_s);

            std::vector<cd> psi_d(nd_, 0.0);
            psi_d[delay_mode] = 1.0;
            if (delay_mode > 0) psi_d[delay_mode - 1] = 0.10;
            if (delay_mode + 1 < nd_) psi_d[delay_mode + 1] = 0.08;
            normalize_vec(psi_d);
            pure_to_density(psi_d, cell.rho_d);

            renormalize_density(cell.rho_s);
            renormalize_density(cell.rho_d);
        }
    }

    void initialize() {
        initialize_implicate_memory();
        initialize_future_attractor();
        metrics_.clear();
        record_metrics(0);
    }

    void evolve() {
        for (int step = 1; step <= cfg_.steps; ++step) {
            update_active_information(cfg_.dt);
            update_implicate_memory(cfg_.dt);
            update_field_rhythm(cfg_.dt);
            unitary_step(cfg_.dt);
            lindblad_step(cfg_.dt);
            topo_coupling_step(cfg_.dt);
            holorhythm_sync_step(cfg_.dt);
            active_guidance_step(cfg_.dt);
            implicate_reconstruction_step(cfg_.dt);
            weak_measurement_step(cfg_.dt);
            future_attractor_step(cfg_.dt);
            enforce_physicality();
            record_metrics(step);
        }
    }

    void write_timeseries_csv(const std::string& path) const {
        std::ofstream out(path);
        out << "step,field_phase,field_amplitude,field_frequency,phase_lock,moving_whole_coherence,"
               "mean_active_info,mean_implicate_memory,symbol_purity,delay_purity,symbol_coherence,"
               "topo_mean,topo_std,entanglement_proxy,observable_z,observable_semantic\n";
        for (const auto& m : metrics_) {
            out << m.step << ',' << m.field_phase << ',' << m.field_amplitude << ',' << m.field_frequency << ','
                << m.phase_lock << ',' << m.moving_whole_coherence << ',' << m.mean_active_info << ','
                << m.mean_implicate_memory << ',' << m.symbol_purity << ',' << m.delay_purity << ','
                << m.symbol_coherence << ',' << m.topo_mean << ',' << m.topo_std << ','
                << m.entanglement_proxy << ',' << m.observable_z << ',' << m.observable_semantic << '\n';
        }
    }

    std::string summary() const {
        std::ostringstream out;
        out << std::fixed << std::setprecision(4);
        auto sp = global_symbol_populations();
        auto dp = global_delay_populations();

        out << "Chronon Field V8\n================\n";
        out << "Grid: " << rows_ << " x " << cols_ << "\nSteps: " << cfg_.steps << ", dt: " << cfg_.dt << "\n\n";
        out << "Core field metrics:\n";
        out << "  field rhythm phase      = " << field_phase_ << "\n";
        out << "  field rhythm amplitude  = " << field_amplitude_ << "\n";
        out << "  field rhythm frequency  = " << field_frequency_ << "\n";
        out << "  phase-lock score        = " << phase_lock_score() << "\n";
        out << "  moving-whole coherence  = " << moving_whole_coherence() << "\n";
        out << "  mean active info        = " << mean_active_info() << "\n";
        out << "  mean implicate memory   = " << mean_implicate_memory() << "\n";
        out << "  entanglement proxy      = " << entanglement_proxy() << "\n\n";

        out << "Measurement observables:\n";
        out << "  <Z_semantic>            = " << observable_z() << "\n";
        out << "  <SemanticBias>          = " << observable_semantic_bias() << "\n";
        out << "  Attractor agreement     = " << attractor_agreement() << "\n";
        out << "  Holographic overlap     = " << holographic_overlap() << "\n";
        out << "  Reconstruction score    = " << reconstruction_score() << "\n\n";

        out << "Global symbol populations:\n";
        for (int i=0; i<ns_; ++i) out << "  S" << i << " = " << sp[i] << "\n";
        out << "\nGlobal delay populations:\n";
        for (int i=0; i<nd_; ++i) out << "  T" << i << " = " << dp[i] << "\n";

        out << "\nField metrics:\n";
        out << "  mean topo             = " << mean_topo() << "\n";
        out << "  topo std              = " << topo_std() << "\n";
        out << "  symbol purity         = " << avg_symbol_purity() << "\n";
        out << "  delay purity          = " << avg_delay_purity() << "\n";
        out << "  symbol coherence      = " << avg_symbol_coherence() << "\n";
        out << "  inter-cell interface  = " << interface_energy() << "\n";
        out << "  pair correlation      = " << pair_correlation() << "\n";

        out << "\nDominant symbol map:\n";
        for (int r=0; r<rows_; ++r) {
            out << "  ";
            for (int c=0; c<cols_; ++c) out << dominant_symbol(at(r,c)) << ' ';
            out << "\n";
        }

        out << "\nMeasurement map (semantic sign):\n";
        for (int r=0; r<rows_; ++r) {
            out << "  ";
            for (int c=0; c<cols_; ++c) out << semantic_sign(at(r,c)) << ' ';
            out << "\n";
        }

        out << "\nRecent timeseries (last 5 steps):\n";
        int start = std::max(0, (int)metrics_.size() - 5);
        for (int i=start; i<(int)metrics_.size(); ++i) {
            const auto& m = metrics_[i];
            out << "  step " << std::setw(3) << m.step
                << " | lock=" << m.phase_lock
                << " | act=" << m.mean_active_info
                << " | imp=" << m.mean_implicate_memory
                << " | ent=" << m.entanglement_proxy
                << " | Z=" << m.observable_z << "\n";
        }
        return out.str();
    }

private:
    SimConfig cfg_;
    int rows_, cols_, ns_, nd_;
    std::vector<DensityCell> cells_;
    std::vector<double> active_info_, implicate_memory_, future_pull_;
    std::vector<StepMetrics> metrics_;
    std::mt19937 rng_;
    std::normal_distribution<double> gauss_;
    double field_phase_=0.0, field_amplitude_=0.0, field_frequency_=0.35, elapsed_time_=0.0;

    int idx(int r, int c) const { return r * cols_ + c; }
    DensityCell& at(int r, int c) { return cells_[idx(r,c)]; }
    const DensityCell& at(int r, int c) const { return cells_[idx(r,c)]; }

    std::vector<std::pair<int,int>> neighbors(int r, int c) const {
        std::vector<std::pair<int,int>> n;
        if (r > 0) n.push_back({r-1,c});
        if (r + 1 < rows_) n.push_back({r+1,c});
        if (c > 0) n.push_back({r,c-1});
        if (c + 1 < cols_) n.push_back({r,c+1});
        return n;
    }

    static void zero_matrix(std::vector<std::vector<cd>>& M) {
        for (auto &row : M) for (auto &x : row) x = 0.0;
    }
    static void normalize_vec(std::vector<cd>& v) {
        double n=0.0; for (auto &x : v) n += std::norm(x);
        if (n > 1e-12) { double inv = 1.0 / std::sqrt(n); for (auto &x : v) x *= inv; }
    }
    static void pure_to_density(const std::vector<cd>& psi, std::vector<std::vector<cd>>& rho) {
        int n = (int)psi.size();
        for (int i=0;i<n;++i) for (int j=0;j<n;++j) rho[i][j] = psi[i] * std::conj(psi[j]);
    }
    static std::vector<std::vector<cd>> matmul(const std::vector<std::vector<cd>>& A, const std::vector<std::vector<cd>>& B) {
        int n = (int)A.size();
        std::vector<std::vector<cd>> C(n, std::vector<cd>(n, 0.0));
        for (int i=0;i<n;++i) for (int k=0;k<n;++k) for (int j=0;j<n;++j) C[i][j] += A[i][k] * B[k][j];
        return C;
    }
    static double trace_real(const std::vector<std::vector<cd>>& A) {
        double t=0.0; for (int i=0;i<(int)A.size();++i) t += A[i][i].real(); return t;
    }
    static double purity(const std::vector<std::vector<cd>>& rho) { return trace_real(matmul(rho, rho)); }
    static double coherence_l1(const std::vector<std::vector<cd>>& rho) {
        double s=0.0; int n=(int)rho.size();
        for (int i=0;i<n;++i) for (int j=0;j<n;++j) if (i != j) s += std::abs(rho[i][j]);
        return s;
    }
    static void renormalize_density(std::vector<std::vector<cd>>& rho) {
        double tr = trace_real(rho);
        if (tr > 1e-12) for (auto &row : rho) for (auto &x : row) x /= tr;
        int n=(int)rho.size();
        for (int i=0;i<n;++i) {
            rho[i][i] = cd(std::max(0.0, rho[i][i].real()), 0.0);
            for (int j=i+1;j<n;++j) {
                cd avg = 0.5 * (rho[i][j] + std::conj(rho[j][i]));
                rho[i][j] = avg; rho[j][i] = std::conj(avg);
            }
        }
    }
    static double wrap(double a) {
        while (a > M_PI) a -= 2*M_PI;
        while (a < -M_PI) a += 2*M_PI;
        return a;
    }

    int dominant_symbol(const DensityCell& x) const {
        int best=0; double bestv=-1.0;
        for (int i=0;i<ns_;++i) {
            double v = x.rho_s[i][i].real();
            if (v > bestv) { bestv = v; best = i; }
        }
        return best;
    }
    int semantic_sign(const DensityCell& x) const {
        double z = (x.rho_s[0][0].real() + x.rho_s[1][1].real()) - (x.rho_s[2][2].real() + x.rho_s[3][3].real());
        return z >= 0.0 ? 1 : -1;
    }

    double local_symbol_entropy(const DensityCell& x) const {
        double h=0.0;
        for (int i=0;i<ns_;++i) { double p = std::max(1e-12, x.rho_s[i][i].real()); h -= p * std::log(p); }
        return h;
    }
    double local_delay_entropy(const DensityCell& x) const {
        double h=0.0;
        for (int i=0;i<nd_;++i) { double p = std::max(1e-12, x.rho_d[i][i].real()); h -= p * std::log(p); }
        return h;
    }

    std::vector<double> global_symbol_populations() const {
        std::vector<double> p(ns_, 0.0);
        for (const auto &x : cells_) for (int i=0;i<ns_;++i) p[i] += x.rho_s[i][i].real();
        double s = std::accumulate(p.begin(), p.end(), 0.0);
        if (s > 1e-12) for (auto &v : p) v /= s;
        return p;
    }
    std::vector<double> global_delay_populations() const {
        std::vector<double> p(nd_, 0.0);
        for (const auto &x : cells_) for (int i=0;i<nd_;++i) p[i] += x.rho_d[i][i].real();
        double s = std::accumulate(p.begin(), p.end(), 0.0);
        if (s > 1e-12) for (auto &v : p) v /= s;
        return p;
    }

    void initialize_implicate_memory() {
        auto gp = global_symbol_populations();
        double g=0.0; for (int i=0;i<ns_;++i) g += (i+1) * gp[i];
        g /= std::max(1, ns_);
        for (auto &m : implicate_memory_) m = g;
    }
    void initialize_future_attractor() {
        for (int r=0;r<rows_;++r) for (int c=0;c<cols_;++c) {
            double x = double(c) / std::max(1, cols_-1);
            double y = double(r) / std::max(1, rows_-1);
            future_pull_[idx(r,c)] = 0.65*(1.0-x)*(1.0-y) + 0.35*x*y;
        }
    }

    void update_active_information(double dt) {
        std::vector<double> next = active_info_;
        for (int r=0;r<rows_;++r) for (int c=0;c<cols_;++c) {
            const auto &x = at(r,c);
            double symbol_entropy = local_symbol_entropy(x);
            double delay_entropy = local_delay_entropy(x);
            double coherence = coherence_l1(x.rho_s);
            double topo_term = std::abs(x.topo - 1.0);
            double semantic_bias = x.rho_s[0][0].real() + 0.7*x.rho_s[1][1].real() - 0.45*x.rho_s[2][2].real() - 0.25*x.rho_s[3][3].real();
            double local = 0.92*coherence - 0.55*symbol_entropy - 0.20*delay_entropy - 0.35*topo_term
                         + 0.32*semantic_bias + 0.12*implicate_memory_[idx(r,c)] + 0.08*future_pull_[idx(r,c)];

            double neighbor_mean=0.0;
            auto nbs = neighbors(r,c);
            for (auto [nr,nc] : nbs) neighbor_mean += active_info_[idx(nr,nc)];
            if (!nbs.empty()) neighbor_mean /= nbs.size();
            next[idx(r,c)] += dt * (-0.10*active_info_[idx(r,c)] + 0.24*local + 0.18*neighbor_mean);
        }
        active_info_.swap(next);
    }

    void update_implicate_memory(double dt) {
        auto gs = global_symbol_populations();
        auto gd = global_delay_populations();
        double global_signature=0.0;
        for (int i=0;i<ns_;++i) global_signature += (i+1) * gs[i];
        for (int i=0;i<nd_;++i) global_signature += 0.7 * (i+1) * gd[i];
        global_signature /= (ns_ + 0.7*nd_);

        std::vector<double> next = implicate_memory_;
        for (int r=0;r<rows_;++r) for (int c=0;c<cols_;++c) {
            const auto &x = at(r,c);
            double local_signature=0.0;
            for (int i=0;i<ns_;++i) local_signature += (i+1)*x.rho_s[i][i].real();
            for (int i=0;i<nd_;++i) local_signature += 0.7*(i+1)*x.rho_d[i][i].real();
            local_signature /= (ns_ + 0.7*nd_);
            double neighbor_mean=0.0;
            auto nbs=neighbors(r,c);
            for (auto [nr,nc] : nbs) neighbor_mean += implicate_memory_[idx(nr,nc)];
            if (!nbs.empty()) neighbor_mean /= nbs.size();

            double lock = 0.5 + 0.5 * std::cos(wrap(x.phase_bias - field_phase_));
            double target = 0.55*global_signature + 0.25*local_signature + 0.20*neighbor_mean;
            next[idx(r,c)] += dt * (0.75*lock) * (target - implicate_memory_[idx(r,c)]);
        }
        implicate_memory_.swap(next);
    }

    void update_field_rhythm(double dt) {
        elapsed_time_ += dt;
        double xsum=0.0, ysum=0.0, amp=0.0;
        for (int r=0;r<rows_;++r) for (int c=0;c<cols_;++c) {
            const auto &cell = at(r,c);
            double local_weight = cell.topo + 0.30*active_info_[idx(r,c)] + 0.22*implicate_memory_[idx(r,c)] + 0.08*future_pull_[idx(r,c)];
            local_weight += 0.5 * cell.rho_s[0][1].real();
            xsum += local_weight * std::cos(cell.phase_bias);
            ysum += local_weight * std::sin(cell.phase_bias);
            amp += std::abs(local_weight);
        }
        field_phase_ = std::atan2(ysum, xsum);
        field_amplitude_ = amp / std::max(1, (int)cells_.size());
        field_frequency_ = 0.25 + 0.04*std::tanh(avg_symbol_coherence()/20.0) + 0.02*std::sin(field_phase_)
                                 + 0.01*std::tanh(mean_active_info()) + 0.01*std::tanh(mean_implicate_memory());
    }

    void apply_commutator_step(std::vector<std::vector<cd>>& rho, const std::vector<std::vector<cd>>& H, double dt) const {
        auto Hr = matmul(H, rho), rH = matmul(rho, H);
        int n = (int)rho.size();
        for (int i=0;i<n;++i) for (int j=0;j<n;++j) rho[i][j] += dt * cd(0.0,-1.0) * (Hr[i][j] - rH[i][j]);
    }

    void unitary_step(double dt) {
        auto next = cells_;
        for (int r=0;r<rows_;++r) for (int c=0;c<cols_;++c) {
            const auto &x = at(r,c);
            auto &y = next[idx(r,c)];
            double ai = active_info_[idx(r,c)];
            double im = implicate_memory_[idx(r,c)];
            double fp = future_pull_[idx(r,c)];
            double travel = 0.30*c - 0.22*r;
            double rhythm_phase = field_phase_ + field_frequency_*elapsed_time_ + travel + 0.10*ai + 0.08*im;
            double guidance = std::tanh(ai + 0.4*im + 0.2*fp);

            std::vector<std::vector<cd>> Hs(ns_, std::vector<cd>(ns_, 0.0)), Hd(nd_, std::vector<cd>(nd_, 0.0));
            for (int i=0;i<ns_;++i) {
                double semantic_potential = guidance * ((i == 0 || i == 1) ? -0.06 : 0.05);
                Hs[i][i] = 0.11*i + 0.03*x.topo*(i+1) + 0.018*field_amplitude_*std::cos(rhythm_phase + i)
                         + semantic_potential + 0.018*im*((i<2) ? -1.0 : 0.6)
                         + cfg_.future_attractor_bias * fp * ((i == 0 || i == 1) ? -1.0 : 0.7);
                if (i+1<ns_) Hs[i][i+1] = Hs[i+1][i] = 0.075 + 0.010*std::sin(rhythm_phase) + 0.008*guidance + 0.003*im;
            }
            for (int i=0;i<nd_;++i) {
                Hd[i][i] = 0.10*i + 0.014*field_amplitude_*std::cos(rhythm_phase - i)
                         - 0.010*guidance*(i == 1 ? 1.0 : -0.4) - 0.009*im*(i == 2 ? 1.0 : -0.25);
                if (i+1<nd_) Hd[i][i+1] = Hd[i+1][i] = 0.048 + 0.007*std::cos(rhythm_phase) + 0.003*im;
            }

            y.rho_s = x.rho_s;
            y.rho_d = x.rho_d;
            apply_commutator_step(y.rho_s, Hs, dt);
            apply_commutator_step(y.rho_d, Hd, dt);

            for (auto [nr,nc] : neighbors(r,c)) {
                const auto &nb = at(nr,nc);
                double dphi = nb.phase_bias - x.phase_bias;
                double wave_alignment = std::cos(rhythm_phase - nb.phase_bias);
                double info_gradient = active_info_[idx(nr,nc)] - ai;
                double memory_gradient = implicate_memory_[idx(nr,nc)] - im;
                cd phase = std::polar(1.0, 0.13*dphi + 0.09*wave_alignment + 0.07*info_gradient + 0.05*memory_gradient);

                for (int i=0;i<ns_;++i) for (int j=0;j<ns_;++j) {
                    cd target = phase * nb.rho_s[i][j] * std::conj(phase);
                    y.rho_s[i][j] += dt * (0.095 + 0.018*wave_alignment + 0.012*std::tanh(info_gradient) + 0.008*std::tanh(memory_gradient))
                                   * (target - x.rho_s[i][j]);
                }
                for (int i=0;i<nd_;++i) for (int j=0;j<nd_;++j) {
                    cd target = phase * nb.rho_d[i][j] * std::conj(phase);
                    y.rho_d[i][j] += dt * (0.060 + 0.012*wave_alignment + 0.007*std::tanh(memory_gradient))
                                   * (target - x.rho_d[i][j]);
                }
                y.topo += dt * 0.15 * (1.0 + 0.08*wave_alignment + 0.05*std::tanh(info_gradient)) * (nb.topo - x.topo);
            }
            y.phase_bias = x.phase_bias + dt * (0.024*x.topo + 0.10*field_amplitude_*std::sin(rhythm_phase) + 0.045*guidance + 0.025*im);
        }
        cells_.swap(next);
    }

    void lindblad_step(double dt) {
        for (int r=0;r<rows_;++r) for (int c=0;c<cols_;++c) {
            auto &x = at(r,c);
            double ai = active_info_[idx(r,c)];
            double im = implicate_memory_[idx(r,c)];
            double lock = std::cos(x.phase_bias - field_phase_);
            double dephase_scale = (1.0 - 0.35*std::max(0.0, lock)) * (1.0 - 0.12*std::tanh(std::max(0.0, ai))) * (1.0 - 0.08*std::tanh(std::max(0.0, im)));

            for (int i=0;i<ns_;++i) for (int j=0;j<ns_;++j) if (i != j) x.rho_s[i][j] *= std::exp(-dephase_scale * 0.016 * std::abs(i-j) * dt);
            for (int high=1; high<ns_; ++high) {
                double moved = dt * 0.018 * high * std::max(0.0, x.rho_s[high][high].real());
                x.rho_s[high][high] -= moved; x.rho_s[high-1][high-1] += moved;
            }

            for (int i=0;i<nd_;++i) for (int j=0;j<nd_;++j) if (i != j) x.rho_d[i][j] *= std::exp(-dephase_scale * 0.013 * std::abs(i-j) * dt);
            for (int low=0; low+1<nd_; ++low) {
                double moved = dt * 0.009 * (low+1) * std::max(0.0, x.rho_d[low][low].real());
                x.rho_d[low][low] -= moved; x.rho_d[low+1][low+1] += moved;
            }

            for (int i=0;i<ns_;++i) x.rho_s[i][i] += 0.00035*std::sqrt(dt)*gauss_(rng_);
            for (int i=0;i<nd_;++i) x.rho_d[i][i] += 0.00025*std::sqrt(dt)*gauss_(rng_);
            x.topo += 0.0025*std::sqrt(dt)*gauss_(rng_);
        }
    }

    void topo_coupling_step(double dt) {
        for (int r=0;r<rows_;++r) for (int c=0;c<cols_;++c) {
            auto &x = at(r,c);
            double ai = active_info_[idx(r,c)];
            double im = implicate_memory_[idx(r,c)];
            x.topo += dt * (0.80 + 0.05*std::tanh(ai) + 0.03*std::tanh(im)) * (1.0 - x.topo);
            for (int i=0;i<ns_;++i) for (int j=i+1;j<ns_;++j) {
                double desired = 0.095*x.topo*(i-j) + 0.04*std::sin(field_phase_) + 0.03*std::tanh(ai) + 0.02*std::tanh(im);
                cd rot = std::polar(1.0, dt*desired);
                x.rho_s[i][j] *= rot; x.rho_s[j][i] = std::conj(x.rho_s[i][j]);
            }
        }
    }

    void holorhythm_sync_step(double dt) {
        for (int r=0;r<rows_;++r) for (int c=0;c<cols_;++c) {
            auto &x = at(r,c);
            double ai = active_info_[idx(r,c)];
            double im = implicate_memory_[idx(r,c)];
            double delta = wrap(field_phase_ - x.phase_bias);
            double sync_strength = dt * (0.06 + 0.08*field_amplitude_ + 0.03*std::tanh(std::max(0.0, ai)) + 0.02*std::tanh(std::max(0.0, im)));
            x.phase_bias += sync_strength * std::sin(delta);
            double align = 0.5*(1.0 + std::cos(delta));
            for (int i=0;i<ns_;++i) for (int j=0;j<ns_;++j) if (i != j) x.rho_s[i][j] *= (1.0 + dt*(0.012*align + 0.004*std::max(0.0, im)));
        }
    }

    void active_guidance_step(double dt) {
        for (int r=0;r<rows_;++r) for (int c=0;c<cols_;++c) {
            auto &x = at(r,c);
            double ai = active_info_[idx(r,c)];
            double im = implicate_memory_[idx(r,c)];
            double pull01 = dt * 0.018 * std::tanh(ai + 0.3*im);
            if (pull01 > 0.0) {
                double moved2 = pull01 * std::max(0.0, x.rho_s[2][2].real());
                double moved3 = pull01 * std::max(0.0, x.rho_s[3][3].real());
                x.rho_s[2][2] -= moved2; x.rho_s[3][3] -= moved3;
                x.rho_s[1][1] += 0.60*moved2 + 0.45*moved3;
                x.rho_s[0][0] += 0.40*moved2 + 0.55*moved3;
            }
        }
    }

    void implicate_reconstruction_step(double dt) {
        auto gs = global_symbol_populations();
        auto gd = global_delay_populations();
        for (int r=0;r<rows_;++r) for (int c=0;c<cols_;++c) {
            auto &x = at(r,c);
            double im = implicate_memory_[idx(r,c)];
            double rs = dt * cfg_.repair_strength * std::max(0.0, im);
            for (int i=0;i<ns_;++i) x.rho_s[i][i] += rs * (gs[i] - x.rho_s[i][i].real());
            for (int i=0;i<nd_;++i) x.rho_d[i][i] += 0.75*rs * (gd[i] - x.rho_d[i][i].real());
        }
    }

    void weak_measurement_step(double dt) {
        for (auto &x : cells_) {
            int dom = dominant_symbol(x);
            for (int i=0;i<ns_;++i) {
                double factor = (i == dom) ? (1.0 + cfg_.measurement_strength*dt) : (1.0 - 0.35*cfg_.measurement_strength*dt);
                x.rho_s[i][i] *= factor;
            }
            for (int i=0;i<ns_;++i) for (int j=0;j<ns_;++j) if (i != j) x.rho_s[i][j] *= (1.0 - 0.12*cfg_.measurement_strength*dt);
        }
    }

    void future_attractor_step(double dt) {
        for (int r=0;r<rows_;++r) for (int c=0;c<cols_;++c) {
            auto &x = at(r,c);
            double fp = future_pull_[idx(r,c)];
            double b = cfg_.future_attractor_bias * dt * fp;
            double moved3 = b * x.rho_s[3][3].real();
            double moved2 = 0.7 * b * x.rho_s[2][2].real();
            x.rho_s[3][3] -= moved3; x.rho_s[2][2] -= moved2;
            x.rho_s[1][1] += 0.55*moved3 + 0.60*moved2;
            x.rho_s[0][0] += 0.45*moved3 + 0.40*moved2;
            x.topo += 0.25*b * (1.0 - x.topo);
        }
    }

    void enforce_physicality() {
        for (auto &x : cells_) { renormalize_density(x.rho_s); renormalize_density(x.rho_d); }
    }

    double phase_lock_score() const { double s=0.0; for (const auto &x : cells_) s += std::cos(wrap(x.phase_bias - field_phase_)); return s / cells_.size(); }
    double moving_whole_coherence() const { double s=0.0; for (const auto &x : cells_) s += coherence_l1(x.rho_s) * (0.5 + 0.5*std::cos(wrap(x.phase_bias - field_phase_))); return s / cells_.size(); }
    double mean_active_info() const { return std::accumulate(active_info_.begin(), active_info_.end(), 0.0) / active_info_.size(); }
    double mean_implicate_memory() const { return std::accumulate(implicate_memory_.begin(), implicate_memory_.end(), 0.0) / implicate_memory_.size(); }
    double mean_topo() const { double s=0.0; for (const auto &x : cells_) s += x.topo; return s / cells_.size(); }
    double topo_std() const { double mu=mean_topo(), s=0.0; for (const auto &x : cells_) { double d=x.topo-mu; s += d*d; } return std::sqrt(s / cells_.size()); }
    double avg_symbol_purity() const { double s=0.0; for (const auto &x : cells_) s += purity(x.rho_s); return s / cells_.size(); }
    double avg_delay_purity() const { double s=0.0; for (const auto &x : cells_) s += purity(x.rho_d); return s / cells_.size(); }
    double avg_symbol_coherence() const { double s=0.0; for (const auto &x : cells_) s += coherence_l1(x.rho_s); return s / cells_.size(); }
    double holographic_overlap() const {
        double mu = mean_implicate_memory(), s=0.0;
        for (double x : implicate_memory_) s += std::exp(-std::abs(x - mu));
        return s / implicate_memory_.size();
    }
    double reconstruction_score() const {
        auto gs = global_symbol_populations();
        double s=0.0;
        for (const auto &x : cells_) {
            double mismatch=0.0;
            for (int i=0;i<ns_;++i) mismatch += std::abs(x.rho_s[i][i].real() - gs[i]);
            s += std::exp(-mismatch);
        }
        return s / cells_.size();
    }
    double interface_energy() const {
        double e=0.0;
        for (int r=0;r<rows_;++r) for (int c=0;c<cols_;++c) {
            const auto &x = at(r,c);
            for (auto [nr,nc] : neighbors(r,c)) {
                const auto &y = at(nr,nc);
                for (int i=0;i<ns_;++i) e += std::abs(x.rho_s[i][i].real() - y.rho_s[i][i].real());
            }
        }
        return e;
    }
    double pair_correlation() const {
        double s=0.0; int count=0;
        for (int r=0;r<rows_;++r) for (int c=0;c<cols_;++c) {
            double zi = semantic_sign(at(r,c));
            for (auto [nr,nc] : neighbors(r,c)) { s += zi * semantic_sign(at(nr,nc)); count++; }
        }
        return count ? s / count : 0.0;
    }
    double entanglement_proxy() const { double s=0.0; for (const auto &x : cells_) s += (1.0 - purity(x.rho_s)); return s / cells_.size(); }
    double observable_z() const {
        double s=0.0;
        for (const auto &x : cells_) s += (x.rho_s[0][0].real() + x.rho_s[1][1].real()) - (x.rho_s[2][2].real() + x.rho_s[3][3].real());
        return s / cells_.size();
    }
    double observable_semantic_bias() const {
        double s=0.0;
        for (const auto &x : cells_) s += x.rho_s[0][0].real() + 0.5*x.rho_s[1][1].real() - 0.5*x.rho_s[2][2].real() - x.rho_s[3][3].real();
        return s / cells_.size();
    }
    double attractor_agreement() const {
        double s=0.0;
        for (int r=0;r<rows_;++r) for (int c=0;c<cols_;++c) {
            const auto &x = at(r,c);
            double semantic = x.rho_s[0][0].real() + x.rho_s[1][1].real() - x.rho_s[2][2].real() - x.rho_s[3][3].real();
            s += semantic * future_pull_[idx(r,c)];
        }
        return s / cells_.size();
    }

    void record_metrics(int step) {
        StepMetrics m;
        m.step = step;
        m.field_phase = field_phase_;
        m.field_amplitude = field_amplitude_;
        m.field_frequency = field_frequency_;
        m.phase_lock = phase_lock_score();
        m.moving_whole_coherence = moving_whole_coherence();
        m.mean_active_info = mean_active_info();
        m.mean_implicate_memory = mean_implicate_memory();
        m.symbol_purity = avg_symbol_purity();
        m.delay_purity = avg_delay_purity();
        m.symbol_coherence = avg_symbol_coherence();
        m.topo_mean = mean_topo();
        m.topo_std = topo_std();
        m.entanglement_proxy = entanglement_proxy();
        m.observable_z = observable_z();
        m.observable_semantic = observable_semantic_bias();
        metrics_.push_back(m);
    }
};

int main() {
    SimConfig cfg;
    ChrononFieldV8 sim(cfg);
    sim.seed_region(0, 3, 0, 7, 0, 2, 0.42,  0.14, 0.04, 1, 0.72);
    sim.seed_region(4, 7, 0, 7, 1, 3, 0.36, -0.08, 0.12, 2, 1.22);
    sim.initialize();
    sim.evolve();
    sim.write_timeseries_csv("/mnt/data/chronon_field_v8_timeseries.csv");
    std::cout << sim.summary();
    return 0;
}
```
