// @generated by protobuf-ts 2.9.4
// @generated from protobuf file "solver/v1/solver.proto" (package "solver.v1", syntax proto3)
// tslint:disable
import type { RpcTransport } from "@protobuf-ts/runtime-rpc";
import type { ServiceInfo } from "@protobuf-ts/runtime-rpc";
import { Solver } from "./solver";
import type { SolverStepByStepResponse } from "./solver";
import { stackIntercept } from "@protobuf-ts/runtime-rpc";
import type { SolverSolveResponse } from "./solver";
import type { SolverSolveRequest } from "./solver";
import type { ServerStreamingCall } from "@protobuf-ts/runtime-rpc";
import type { RpcOptions } from "@protobuf-ts/runtime-rpc";
/**
 * @generated from protobuf service solver.v1.Solver
 */
export interface ISolverClient {
    /**
     * streams the data in response
     *
     * @generated from protobuf rpc: Solve(solver.v1.SolverSolveRequest) returns (stream solver.v1.SolverSolveResponse);
     */
    solve(input: SolverSolveRequest, options?: RpcOptions): ServerStreamingCall<SolverSolveRequest, SolverSolveResponse>;
    /**
     * @generated from protobuf rpc: SolveStepByStep(solver.v1.SolverSolveRequest) returns (stream solver.v1.SolverStepByStepResponse);
     */
    solveStepByStep(input: SolverSolveRequest, options?: RpcOptions): ServerStreamingCall<SolverSolveRequest, SolverStepByStepResponse>;
}
/**
 * @generated from protobuf service solver.v1.Solver
 */
export class SolverClient implements ISolverClient, ServiceInfo {
    typeName = Solver.typeName;
    methods = Solver.methods;
    options = Solver.options;
    constructor(private readonly _transport: RpcTransport) {
    }
    /**
     * streams the data in response
     *
     * @generated from protobuf rpc: Solve(solver.v1.SolverSolveRequest) returns (stream solver.v1.SolverSolveResponse);
     */
    solve(input: SolverSolveRequest, options?: RpcOptions): ServerStreamingCall<SolverSolveRequest, SolverSolveResponse> {
        const method = this.methods[0], opt = this._transport.mergeOptions(options);
        return stackIntercept<SolverSolveRequest, SolverSolveResponse>("serverStreaming", this._transport, method, opt, input);
    }
    /**
     * @generated from protobuf rpc: SolveStepByStep(solver.v1.SolverSolveRequest) returns (stream solver.v1.SolverStepByStepResponse);
     */
    solveStepByStep(input: SolverSolveRequest, options?: RpcOptions): ServerStreamingCall<SolverSolveRequest, SolverStepByStepResponse> {
        const method = this.methods[1], opt = this._transport.mergeOptions(options);
        return stackIntercept<SolverSolveRequest, SolverStepByStepResponse>("serverStreaming", this._transport, method, opt, input);
    }
}
