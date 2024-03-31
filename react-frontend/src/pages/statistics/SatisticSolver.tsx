import {Grid} from "@mui/material";
import {Piece, SolverSolveResponse} from "../../proto/solver/v1/solver.ts";
import {RequestFormStatistics} from "./RequestFormStatistics.tsx";
import {useRecoilState, useRecoilValue} from "recoil";
import {Board} from "../requestForm/atoms.ts";
import {default as BoardComponent} from "../../components/Board.tsx";
import {useEffect, useState} from "react";
import {GrpcWebFetchTransport} from "@protobuf-ts/grpcweb-transport";
import {abortController, SERVER_BASE_URL} from "../../utils/Constants.tsx";
import {SolverClient} from "../../proto/solver/v1/solver.client.ts";
import {generatedBoardsState, isSolvingStatisticsState, settingsStatisticsState} from "./atoms.ts";

export const SatisticSolver = () => {

    const setting = useRecoilValue(settingsStatisticsState);
    const [solving, setSolving] = useState(true);
    const [solvingStat, setSolvingStat] = useRecoilState(isSolvingStatisticsState);
    const [startedStatistics, setStartedStatistics] = useState(false);
    const [startedSolving, setStartedSolving] = useState(false);
    const [responses, setResponses] = useState<{
        [key: string]: {
            board: Board,
            response: SolverSolveResponse
        }

    }>({});
    const [currentGeneratedBoardIndex, setCurrentGeneratedBoardIndex] = useState(0);

    const generatedBoards = useRecoilValue(generatedBoardsState);
    const [waiting, setWaiting] = useState(false);

    useEffect(() => {
            if ((solvingStat && !startedStatistics) || waiting) {
                setStartedStatistics(true);
                console.log("started statistics");
                const generatedBoard = generatedBoards[currentGeneratedBoardIndex];
                if (solving && !startedSolving) {
                    setStartedSolving(true);
                    console.log("started solving statistics" , generatedBoard.label);

                    const transport = new GrpcWebFetchTransport({
                        baseUrl: SERVER_BASE_URL,
                        format: "binary",
                        abort: abortController.abortController.signal,

                    });

                    const solverClient = new SolverClient(
                        transport
                    );
                    const stream = solverClient.solve({
                        "hashThreshold": setting.hashThreshold,
                        "pieces": generatedBoard.pieces.map(piece => piece.piece as Piece),
                        "threads": setting.threads,
                        "waitTime": setting.waitTime,
                        solvePath: [],
                        useCache: setting.useCache,
                        cachePullInterval: setting.cachePullInterval
                    }, {});
                    stream.responses.onMessage((message) => {
                            setResponses((prev) => {
                                    return {
                                        ...prev,
                                        [generatedBoard.label]: {
                                            board: generatedBoard,
                                            response: message
                                        }

                                    }
                                }
                            );
                        }
                    );
                    stream.responses.onError(() => {
                            setSolving(true);
                            setStartedSolving(false);
                        }
                    );
                    
                    stream.responses.onComplete(() => {
                        setSolving(true);
                        setStartedSolving(false);
                        setCurrentGeneratedBoardIndex((prev) => prev + 1);
                        console.log("Completed solving statistics", generatedBoard.label);
                        console.log("currentGeneratedBoardIndex", currentGeneratedBoardIndex);

                        if (currentGeneratedBoardIndex < generatedBoards.length - 1) {
                            setWaiting(true);
                        } else {
                            setWaiting(false);
                        }
                    });


                }
            }

        }

        , [solvingStat, startedStatistics, startedSolving, generatedBoards, setting, responses, setResponses, setSolvingStat, currentGeneratedBoardIndex, solving, waiting]);

    useEffect(() => {
        return () => {
            abortController.abortController.abort();
            abortController.abortController = new AbortController();
        }
    }, []);

    console.log("responses", responses);
    return (
        <>
            <Grid container spacing={2}
                  style={{
                      minHeight: "90vh",
                      height: '100%',
                      display: 'flex',
                      justifyContent: 'center',
                      alignItems: 'center'
                  }}>

                <Grid item xs={5}>
                    <div
                        style={{
                            display: "flex",
                            flexDirection: "column",
                            alignItems: "center",
                            justifyContent: "center",
                            width: "100%",
                            margin: "auto",
                        }}
                    >
                        <RequestFormStatistics/>
                    </div>
                </Grid>
                <Grid item xs={4}>
                    <div style={{
                        display: 'flex',
                        justifyContent: 'center',
                        alignItems: 'center',
                        height: '100%',
                        aspectRatio: 1,
                    }}>

                    </div>
                </Grid>
            </Grid>

        </>
    )
}