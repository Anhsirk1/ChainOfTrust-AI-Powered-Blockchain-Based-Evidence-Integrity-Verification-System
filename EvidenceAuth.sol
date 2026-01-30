// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract EvidenceAuth {
    struct LogEntry {
        address actor;
        bytes32 dataHash;
        string role;
        string action;
        uint256 timestamp;
    }

    LogEntry[] public logs;
    event LogRecorded(address indexed actor, bytes32 dataHash, string role, string action, uint256 timestamp);

    function recordEvent(bytes32 _dataHash, string memory _role, string memory _action) public {
        uint256 ts = block.timestamp;
        logs.push(LogEntry(msg.sender, _dataHash, _role, _action, ts));
        emit LogRecorded(msg.sender, _dataHash, _role, _action, ts);
    }

    function getLogsCount() public view returns (uint256) {
        return logs.length;
    }

    function getLog(uint256 index) public view returns (address, bytes32, string memory, string memory, uint256) {
        require(index < logs.length, "Invalid index");
        LogEntry memory e = logs[index];
        return (e.actor, e.dataHash, e.role, e.action, e.timestamp);
    }
}
