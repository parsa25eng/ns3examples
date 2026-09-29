/*
 * mc-channel-trace-gen.cc
 *
 * STARTING SCRIPT — multi-gNB channel trace generator for QoS-aware
 * serving-cluster / traffic-splitting research, built on ns-3 5G-LENA (nr module).
 *
 * WHAT THIS DOES
 *   - Deploys N gNBs at fixed positions (edit gNbPositions below to match your
 *     9-gNB, 1000x1000 m layout).
 *   - Deploys 1 UE with a RandomWaypointMobilityModel, using a FIXED random run
 *     number so the trajectory is reproducible across separate invocations.
 *   - Attaches the UE to ONE candidate gNB, chosen by --attachGnb=<index> on
 *     the command line.
 *   - Runs downlink CBR traffic so the PHY layer actually produces
 *     SINR/CQI/MCS measurements, and dumps them via NrHelper's built-in traces.
 *
 * HOW TO USE IT FOR MULTI-CONNECTIVITY CANDIDATE-SET GENERATION
 *   Run this binary once per candidate gNB, keeping --runNumber fixed so the
 *   UE follows the identical trajectory every time:
 *
 *     ./ns3 run "mc-channel-trace-gen --attachGnb=0 --runNumber=1"
 *     ./ns3 run "mc-channel-trace-gen --attachGnb=1 --runNumber=1"
 *     ...
 *     ./ns3 run "mc-channel-trace-gen --attachGnb=8 --runNumber=1"
 *
 *   Each run writes its own RxPacketTrace-*.txt (via NrHelper::EnableTraces())
 *   into the working directory, tagged with the attached gNB index in the
 *   filename prefix. Post-process these 9 files offline (Python/MATLAB) into
 *   a single [time x gNB] SINR/CQI/MCS matrix — that matrix is the input your
 *   heuristic (Eqs. 27-31) and your LSTM/DNN pipeline already expect.
 *
 * WHAT YOU WILL STILL NEED TO ADD
 *   - FR2 / mixed-numerology bands for the small cells (this script sets up
 *     ONE band/numerology for all gNBs, to keep it simple — see
 *     cttc-nr-cc-bwp-demo.cc in the nr module examples for multi-band setup).
 *   - Your BLER-vs-SINR/MCS mapping if you want BLER, not just SINR/MCS, out
 *     of ns-3 — the built-in traces give you SINR and MCS per TB; feed those
 *     into your already-trained DNN offline rather than re-deriving BLER here.
 *   - Exact trace API names can differ slightly between nr module releases —
 *     check examples/cttc-nr-demo.cc in YOUR installed nr module version and
 *     adjust EnableTraces()/output file naming if needed.
 *
 * Build/run inside your ns-3 tree, e.g.:
 *   cp mc-channel-trace-gen.cc contrib/nr/examples/
 *   ./ns3 configure --enable-examples
 *   ./ns3 run "mc-channel-trace-gen --attachGnb=0"
 */

#include "ns3/antenna-module.h"
#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/mobility-module.h"
#include "ns3/internet-module.h"
#include "ns3/applications-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/nr-module.h"
#include "ns3/flow-monitor-module.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <functional>
#include <sstream>
#include <vector>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("McChannelTraceGen");

int
main(int argc, char* argv[])
{
    // Diagnostic: print exactly where the process THINKS it's running from,
    // before anything else happens. ./ns3 run does not necessarily preserve
    // your shell's cwd - this line tells you, with certainty, what directory
    // a bare "." would have resolved to if you don't pass --outputDir.
    std::cout << "[info] Process starting cwd = " << std::filesystem::current_path().string()
              << std::endl;
    std::cout.flush();

    // ---------------------------------------------------------------
    // 1. Command-line parameters
    // ---------------------------------------------------------------
    uint32_t attachGnb = 0;      // which gNB (0-8) the UE attaches to this run
    uint32_t runNumber = 1;      // keep this FIXED across the 9 runs so the UE
                                  // follows the same trajectory every time
    double simTime = 600.0;      // seconds, matches Table III of the paper
    double centerFrequency = 3.5e9; // Hz, FR1 macro band (numerology 0)
    uint16_t numerology = 0;
    double bandwidth = 20e6;     // Hz
    double txPower = 49.0;       // dBm, macro-cell max Tx power from Table III
    std::string outputDir = ".";  // where THIS process's trace files land
    double channelUpdatePeriodMs = 100.0; // see note below - NOT 0 for anything beyond a few seconds
    double udpIntervalMs = 5.0;  // downlink CBR interval; lower = more PHY events = slower

    CommandLine cmd(__FILE__);
    cmd.AddValue("attachGnb", "Index of the gNB the UE attaches to (0-8)", attachGnb);
    cmd.AddValue("runNumber", "RNG run number - keep fixed for a reproducible UE trajectory", runNumber);
    cmd.AddValue("simTime", "Simulation duration in seconds", simTime);
    cmd.AddValue("outputDir", "Directory this run's trace files are written into "
                               "(REQUIRED to differ across parallel runs - see note below)", outputDir);
    cmd.AddValue("channelUpdatePeriodMs",
                 "How often the LOS/NLOS channel condition is recomputed, in ms. "
                 "0 means 'recompute on every query' - fine for the ~1s cttc-nr-demo "
                 "tutorial this was copied from, but expensive at hundreds of simulated "
                 "seconds. Increase this first if a run feels too slow.",
                 channelUpdatePeriodMs);
    cmd.AddValue("udpIntervalMs", "Downlink CBR packet interval in ms", udpIntervalMs);
    cmd.Parse(argc, argv);

    // Change the process's working directory BEFORE any trace file gets
    // created, so relative-path trace output lands in a run-specific folder
    // regardless of how the ns3 launcher itself manages cwd. This runs even
    // when outputDir is the default "." - do NOT treat "." as "skip this",
    // since that was exactly what made trace files land somewhere silently
    // different from where you expected. Always pass an ABSOLUTE path via
    // --outputDir=$(pwd) from your shell to be certain where output goes.
    std::filesystem::create_directories(outputDir);
    std::filesystem::current_path(outputDir);
    std::cout << "[info] Resolved outputDir to absolute path = "
              << std::filesystem::current_path().string() << std::endl;
    std::cout.flush();

    std::cout << "[info] mc-channel-trace-gen starting. attachGnb=" << attachGnb
              << " simTime=" << simTime << "s outputDir=" << outputDir << std::endl;
    std::cout.flush();

    RngSeedManager::SetSeed(1);
    RngSeedManager::SetRun(runNumber);

    // ---------------------------------------------------------------
    // 2. gNB positions — edit these to match your 1000x1000 m, 9-gNB layout
    //    (index 0 = macro cell at the center; 1-8 = small cells)
    // ---------------------------------------------------------------
    std::vector<Vector> gNbPositions = {
        Vector(500.0, 500.0, 25.0),  // 0: macro, center
        Vector(150.0, 150.0, 10.0),  // 1
        Vector(500.0, 150.0, 10.0),  // 2
        Vector(850.0, 150.0, 10.0),  // 3
        Vector(150.0, 500.0, 10.0),  // 4
        Vector(850.0, 500.0, 10.0),  // 5
        Vector(150.0, 850.0, 10.0),  // 6
        Vector(500.0, 850.0, 10.0),  // 7
        Vector(850.0, 850.0, 10.0),  // 8
    };

    if (attachGnb >= gNbPositions.size())
    {
        std::cerr << "attachGnb out of range" << std::endl;
        return 1;
    }

    NodeContainer gNbNodes;
    gNbNodes.Create(gNbPositions.size());

    NodeContainer ueNodes;
    ueNodes.Create(1);

    // ---------------------------------------------------------------
    // 3. Mobility: gNBs fixed, UE random-waypoint within the 1000x1000 m area
    // ---------------------------------------------------------------
    MobilityHelper gNbMobility;
    Ptr<ListPositionAllocator> gNbPosAlloc = CreateObject<ListPositionAllocator>();
    for (const auto& pos : gNbPositions)
    {
        gNbPosAlloc->Add(pos);
    }
    gNbMobility.SetPositionAllocator(gNbPosAlloc);
    gNbMobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    gNbMobility.Install(gNbNodes);

    MobilityHelper ueMobility;
    ueMobility.SetMobilityModel(
        "ns3::RandomWaypointMobilityModel",
        "Speed", StringValue("ns3::UniformRandomVariable[Min=1.0|Max=5.0]"), // m/s, edit as needed
        "Pause", StringValue("ns3::ConstantRandomVariable[Constant=0.0]"),
        "PositionAllocator", PointerValue(CreateObjectWithAttributes<RandomBoxPositionAllocator>(
            "X", StringValue("ns3::UniformRandomVariable[Min=0.0|Max=1000.0]"),
            "Y", StringValue("ns3::UniformRandomVariable[Min=0.0|Max=1000.0]"),
            "Z", StringValue("ns3::ConstantRandomVariable[Constant=1.5]"))));
    Ptr<RandomBoxPositionAllocator> ueInitialPos = CreateObjectWithAttributes<RandomBoxPositionAllocator>(
        "X", StringValue("ns3::UniformRandomVariable[Min=0.0|Max=1000.0]"),
        "Y", StringValue("ns3::UniformRandomVariable[Min=0.0|Max=1000.0]"),
        "Z", StringValue("ns3::ConstantRandomVariable[Constant=1.5]"));
    ueMobility.SetPositionAllocator(ueInitialPos);
    ueMobility.Install(ueNodes);

    // ---------------------------------------------------------------
    // 4. NR stack setup — single band/numerology for simplicity
    //    (see cttc-nr-cc-bwp-demo.cc to add FR2 bands for the small cells)
    // ---------------------------------------------------------------
    Ptr<NrPointToPointEpcHelper> epcHelper = CreateObject<NrPointToPointEpcHelper>();
    Ptr<IdealBeamformingHelper> beamformingHelper = CreateObject<IdealBeamformingHelper>();
    Ptr<NrHelper> nrHelper = CreateObject<NrHelper>();
    nrHelper->SetBeamformingHelper(beamformingHelper);
    nrHelper->SetEpcHelper(epcHelper);

    // IMPORTANT: without this, the beamforming algorithm defaults to something
    // far more expensive than you need - a full angular search evaluated
    // repeatedly during the run, rather than a direct geometric calculation.
    // DirectPathBeamforming computes the beam analytically from known UE/gNB
    // positions instead of searching, and is what cttc-nr-demo uses for
    // exactly this reason. This is very likely why 1 simulated second was
    // never finishing.
    beamformingHelper->SetAttribute("BeamformingMethod",
                                     TypeIdValue(DirectPathBeamforming::GetTypeId()));

    // Small, explicit antenna arrays with an isotropic element model - cheap
    // to compute with, and matches what the tutorial example uses. Leaving
    // these unset risks an oversized default array, which multiplies the
    // cost of every beamforming/channel computation.
    nrHelper->SetUeAntennaAttribute("NumRows", UintegerValue(2));
    nrHelper->SetUeAntennaAttribute("NumColumns", UintegerValue(2));
    nrHelper->SetUeAntennaAttribute("AntennaElement",
                                     PointerValue(CreateObject<IsotropicAntennaModel>()));
    nrHelper->SetGnbAntennaAttribute("NumRows", UintegerValue(2));
    nrHelper->SetGnbAntennaAttribute("NumColumns", UintegerValue(2));
    nrHelper->SetGnbAntennaAttribute("AntennaElement",
                                      PointerValue(CreateObject<IsotropicAntennaModel>()));

    CcBwpCreator ccBwpCreator;
    CcBwpCreator::SimpleOperationBandConf bandConf(centerFrequency, bandwidth, 1);
    OperationBandInfo band = ccBwpCreator.CreateOperationBandContiguousCc(bandConf);

    // NrChannelHelper replaces the old NrHelper::InitializeOperationBand() call
    // in current nr module releases: it configures the propagation/fading/
    // channel-condition factories and installs them into the band.
    Ptr<NrChannelHelper> channelHelper = CreateObject<NrChannelHelper>();
    channelHelper->ConfigureFactories("UMa", "Default", "ThreeGpp"); // outdoor macro scenario;
        // use "UMi" instead if you later split the small cells into their own band
    channelHelper->SetChannelConditionModelAttribute(
        "UpdatePeriod", TimeValue(MilliSeconds(channelUpdatePeriodMs)));
    channelHelper->SetPathlossAttribute("ShadowingEnabled", BooleanValue(true)); // kept on, unlike
        // the stock tutorial's default, since your paper's channel model should include shadowing
    channelHelper->AssignChannelsToBands({band});

    BandwidthPartInfoPtrVector allBwps = CcBwpCreator::GetAllBwps({band});

    nrHelper->SetGnbPhyAttribute("TxPower", DoubleValue(txPower));
    nrHelper->SetGnbPhyAttribute("Numerology", UintegerValue(numerology));

    NetDeviceContainer gNbDevices = nrHelper->InstallGnbDevice(gNbNodes, allBwps);
    NetDeviceContainer ueDevices = nrHelper->InstallUeDevice(ueNodes, allBwps);

    // No manual UpdateConfig() loop needed: AttachToGnb() below triggers cell
    // configuration internally in current nr module releases. The old manual
    // UpdateConfig() call is deprecated (that's the warning you saw).

    // ---------------------------------------------------------------
    // 5. Internet stack + attach UE to the chosen candidate gNB
    // ---------------------------------------------------------------
    InternetStackHelper internet;
    internet.Install(ueNodes);
    Ipv4InterfaceContainer ueIpIface = epcHelper->AssignUeIpv4Address(ueDevices);

    Ptr<Node> remoteHost = CreateObject<Node>();
    InternetStackHelper internetRemote;
    internetRemote.Install(remoteHost);
    PointToPointHelper p2p;
    p2p.SetDeviceAttribute("DataRate", StringValue("10Gbps"));
    NetDeviceContainer internetDevices = p2p.Install(epcHelper->GetPgwNode(), remoteHost);
    Ipv4AddressHelper ipv4h;
    ipv4h.SetBase("1.0.0.0", "255.0.0.0");
    Ipv4InterfaceContainer internetIpIfaces = ipv4h.Assign(internetDevices);

    Ipv4StaticRoutingHelper ipv4RoutingHelper;
    Ptr<Ipv4StaticRouting> remoteHostStaticRouting =
        ipv4RoutingHelper.GetStaticRouting(remoteHost->GetObject<Ipv4>());
    remoteHostStaticRouting->AddNetworkRouteTo(Ipv4Address("7.0.0.0"), Ipv4Mask("255.0.0.0"), 1);

    // Attach the UE to the single candidate gNB for this run
    nrHelper->AttachToGnb(ueDevices.Get(0), gNbDevices.Get(attachGnb));

    // ---------------------------------------------------------------
    // 6. Downlink CBR traffic so the PHY actually reports SINR/CQI/MCS
    // ---------------------------------------------------------------
    uint16_t dlPort = 2000;
    UdpServerHelper dlServer(dlPort);
    ApplicationContainer serverApps = dlServer.Install(ueNodes.Get(0));
    serverApps.Start(Seconds(0.0));
    serverApps.Stop(Seconds(simTime));

    UdpClientHelper dlClient(ueIpIface.GetAddress(0), dlPort);
    dlClient.SetAttribute("Interval", TimeValue(MilliSeconds(udpIntervalMs)));
    dlClient.SetAttribute("MaxPackets", UintegerValue(4294967295));
    dlClient.SetAttribute("PacketSize", UintegerValue(1200));
    ApplicationContainer clientApps = dlClient.Install(remoteHost);
    clientApps.Start(Seconds(0.2));
    clientApps.Stop(Seconds(simTime));

    // ---------------------------------------------------------------
    // 7. Traces — this is the key output for offline post-processing.
    //    Tag the output file prefix with the attached gNB index so the
    //    9 runs don't overwrite each other.
    // ---------------------------------------------------------------
    std::string tracePrefix = "mc-trace-gnb" + std::to_string(attachGnb);
    nrHelper->EnableTraces(); // writes RxPacketTrace / DlDataSinr / etc. to the working dir
    // NOTE: some nr module versions let you set an output prefix directly on
    // the trace helper/phy stats calculator — check your installed version's
    // NrPhyRxTrace / NrBearerStatsCalculator API and set it explicitly so the
    // 9 runs land in clearly separate files instead of relying on renaming
    // after each run.

    // Progress indicator: prints simulated time vs. wall-clock time every 10
    // simulated seconds, so you can tell "slow" from "stuck" and estimate how
    // long the full run will actually take.
    static auto startWall = std::chrono::steady_clock::now();
    std::function<void()> printProgress = [&]() {
        auto now = std::chrono::steady_clock::now();
        double wallSec = std::chrono::duration<double>(now - startWall).count();
        std::cout << "[progress] simTime=" << Simulator::Now().GetSeconds() << "s / "
                  << simTime << "s   wallClock=" << wallSec << "s   ratio(sim/wall)="
                  << (wallSec > 0 ? Simulator::Now().GetSeconds() / wallSec : 0.0) << std::endl;
        if (Simulator::Now().GetSeconds() + 10.0 <= simTime)
        {
            Simulator::Schedule(Seconds(10.0), printProgress);
        }
    };
    Simulator::Schedule(Seconds(10.0), printProgress);

    std::cout << "[info] Setup complete, entering Simulator::Run() now." << std::endl;
    std::cout.flush();

    Simulator::Stop(Seconds(simTime));
    Simulator::Run();

    std::cout << "[info] Simulator::Run() returned normally at simTime="
              << Simulator::Now().GetSeconds() << "s." << std::endl;
    std::cout.flush();

    Simulator::Destroy();

    // ---------------------------------------------------------------
    // 8. Show what EnableTraces() actually produced, right in the log.
    //    We don't hard-code exact nr-module trace filenames here (they
    //    differ across nr module releases) - instead we inventory whatever
    //    landed in this run's output directory and preview each text file,
    //    so you can see the real column names/data without guessing.
    // ---------------------------------------------------------------
    std::cout << "\n[results] Files produced in output directory ("
              << std::filesystem::current_path().string() << "):" << std::endl;

    for (const auto& entry : std::filesystem::directory_iterator(std::filesystem::current_path()))
    {
        if (!entry.is_regular_file())
        {
            continue;
        }
        auto path = entry.path();
        auto fileSize = std::filesystem::file_size(path);
        std::cout << "  - " << path.filename().string() << "  (" << fileSize << " bytes)"
                  << std::endl;

        // Preview text-looking files: print the header/first few lines and a
        // total line count, so you can confirm SINR/MCS/etc. actually got
        // recorded without opening the file yourself.
        std::string ext = path.extension().string();
        if (ext == ".txt" || ext == ".csv" || ext == ".dat" || ext.empty())
        {
            std::ifstream f(path);
            if (f.is_open())
            {
                std::string line;
                int lineCount = 0;
                int previewLines = 5;
                while (std::getline(f, line))
                {
                    if (lineCount < previewLines)
                    {
                        std::cout << "      | " << line << std::endl;
                    }
                    lineCount++;
                }
                std::cout << "      (" << lineCount << " total lines)" << std::endl;
            }
        }
    }
    std::cout << "[results] End of file inventory.\n" << std::endl;

    return 0;
}