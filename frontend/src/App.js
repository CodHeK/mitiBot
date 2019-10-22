import React, { Component } from 'react';
import { Line } from 'react-chartjs-2';
import dataset from './data.json';
import axios from 'axios';
import './App.css';


class App extends Component {
  constructor(props) {
    super(props);
    this.state = {
      totalPkts: [],
      attackPkts: [],
      total: 0,
      attack: 0,
      btn: 'START',
    };
  }


  run = (e) => {
    let that = this;
    let pervTpkts = [];
    let prevApkts = [];
    let counter = 0;
    let remove = 0, chart;

    function timer() {
      if(counter == 60) {
        that.setState({ totalPkts: [], attackPkts: [],
                      total: 0,
                      attack: 0 });

        counter = 0;

        prevApkts = [];
        pervTpkts = [];
      }
      pervTpkts.push(Math.floor((Math.random() * 100) + 50));
      prevApkts.push(Math.floor((Math.random() * 5) + 1));

      counter += 2;

      that.setState({ totalPkts: pervTpkts, attackPkts: prevApkts,
                      total: that.state.total + Math.floor((Math.random() * 100) + 50),
                      attack: that.state.attack + Math.floor((Math.random() * 5) + 1) });
    }

    if(this.state.btn === 'START') {
      this.setState({ btn: 'STOP' });
      
      chart = setInterval(timer, 1000);
    }
    else {
      this.setState({ btn: 'START' });

      clearInterval(chart);
    }
  }

  render() {
    let labels = [];
    for(let i = 0; i <= 60; i += 2) labels.push(i);
    const data = {
      labels,
      datasets: [
        {
          label: 'Total Packets',
          fill: true,
          lineTension: 0.1,
          backgroundColor: 'rgba(55,81,255,0.1)',
          borderColor: '#3751FF',
          borderCapStyle: 'butt',
          borderDash: [],
          borderDashOffset: 0.0,
          borderJoinStyle: 'miter',
          pointBorderColor: 'rgba(75,192,192,1)',
          pointBackgroundColor: '#fff',
          pointBorderWidth: 1,
          pointHoverRadius: 5,
          pointHoverBackgroundColor: 'rgba(75,192,192,1)',
          pointHoverBorderColor: 'rgba(220,220,220,1)',
          pointHoverBorderWidth: 2,
          pointRadius: 1,
          pointHitRadius: 10,
          data: this.state.totalPkts
        },
        {
          label: 'Attack Packets',
          fill: true,
          lineTension: 0.1,
          backgroundColor: 'rgba(75,152,92, 0.2)',
          borderColor: 'rgba(75,152,92,10)',
          borderCapStyle: 'butt',
          borderDash: [],
          borderDashOffset: 0.0,
          borderJoinStyle: 'miter',
          pointBorderColor: 'rgba(75,152,92,10)',
          pointBackgroundColor: '#fff',
          pointBorderWidth: 1,
          pointHoverRadius: 5,
          pointHoverBackgroundColor: 'rgba(75,152,92,10)',
          pointHoverBorderColor: 'rgba(220,220,220,1)',
          pointHoverBorderWidth: 2,
          pointRadius: 1,
          pointHitRadius: 10,
          data: this.state.attackPkts
        }
      ]
    };

    return (
      <div className="App">
        <div className="dashboard">
          <div className="navbar">
              <div className="overview">
                Overview
              </div>
              <div className="start-btn">
                <button className="waves-effect waves-light btn start" onClick={(e) => this.run(e)}>{this.state.btn}</button>
              </div>
          </div>

          <div className="stats">
              <div className="left">
                <div className="number">
                    {this.state.attack}
                </div>
                <div className="text">
                    Threats detected
                </div>
              </div>
              <div className="middle">
                /
              </div>
              <div className="right">
                <div className="number">
                    {this.state.total}
                </div>
                <div className="text">
                    Packets Analysed
                </div>
              </div>
          </div>

          <div className="graph">
            <div className="title">
                Todays Trends
            </div>
            <div className="chart">
              <Line data={data} height={70} options={{ animation: false }} redraw />
            </div>
          </div>

          <div className="logs">
              <div className="title">
                  Hits Response
              </div>
              <div className="table">
                  <table>
                    <thead>
                      <tr>
                          <th>Date</th>
                          <th>Time</th>
                          <th>Source IP</th>
                          <th>Threat</th>
                      </tr>
                    </thead>

                    <tbody>
                      <tr>
                        <td>2011/08/10</td>
                        <td>09:46:53</td>
                        <td>212.50.71.17</td>
                        <td>Yes</td>
                      </tr>
                      <tr>
                        <td>2011/08/10</td>
                        <td>09:46:53</td>
                        <td>147.32.84.229</td>
                        <td>No</td>
                      </tr>
                    </tbody>

                  </table>
              </div>
          </div>
        </div>
      </div>
    );
  }
}

export default App;
