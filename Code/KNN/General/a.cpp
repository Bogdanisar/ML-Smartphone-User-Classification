#include <bits/stdc++.h>
#include <algorithm>

using namespace std;

#if 1
    #define pv(x) std::cerr<<#x<<" = "<<(x)<<"; ";std::cerr.flush()
    #define pn std::cerr<<std::endl
#else
    #define pv(x)
    #define pn
#endif

using ll = long long;
using ull = unsigned long long;
using uint = unsigned int;
using ld = long double;
using pii = pair<int,int>;
using pll = pair<ll,ll>;
using pld = pair<ld, ld>;
#define pb push_back
const double PI = 3.14159265358979323846264338327950288419716939937510582097494459230781640628620899862;
const int inf_int = 1e9 + 5;
const ll inf_ll = 1e18 + 5;
const int NMax = 1e3 + 5;
const int dx[] = {-1,0,0,+1}, dy[] = {0,-1,+1,0};
const double EPS = 1e-8;


vector<string> testFiles;
vector<int> dataOfUser[20 + 5];
map<int, int> userOfInput, predictedUserOfInput;
vector<int> trainId, validationId;

struct Point {
    long double coord[3];
    Point(ld _x, ld _y, ld _z) {
        coord[0] = _x;
        coord[1] = _y;
        coord[2] = _z;
    }
};
unordered_map<int, vector<Point>> exampleDataWithId;
unordered_map<int, vector<Point>> trimmedExampleDataWithId;
ld dp[NMax][NMax];

const int splitNum = 448; ////////////////////////////////////////////////////////////////////////////////////////////////////


void getTrainFiles(vector<string>& ans) {    
    const string testFiles = "../train_files.txt";
    ifstream in;

    in.open(testFiles);

    string line;
    while (true) {
        getline(in, line);
        if (in.fail()) {
            break;
        }

        ans.push_back(line);
    }

    in.close();


    for (int i = 0; i < 5; ++i) {
        cout << ans[i] << '\n';
    }
    cout << "\n\n\n";
}

void getUserTestFiles(vector<int>* arr) {
    const string labelsFile = "../../data/train_labels.csv";
    
    FILE* fin = fopen(labelsFile.c_str(), "r");

    assert(fin != nullptr);
    char line[100];

    int id, user;
    fgets(line, sizeof(line), fin);
    while (true) {
        if (fscanf(fin, "%i,%i\n", &id, &user) == EOF) {
            break;
        }
        // cout << id << ' ' << user << '\n';

        arr[user].pb(id);
    }
    // cout << "\n\n\n";

    fclose(fin);
}

void buildMap(vector<int>* dataOfUser, map<int, int>& userOfInput) {
    for (int u = 1; u <= 20; ++u) {
        for (int id : dataOfUser[u]) {
            userOfInput[id] = u;
        }

        // cout << dataOfUser[u].size() << '\n';
    }

    // pv(userOfInput[23999]);pn; //////////////
    // cout << "\n\n\n";
}

void splitTrainValidation(vector<int>* dataOfUser, vector<int>& trainId, vector<int>& validationId) {

    for (int u = 1; u <= 20; ++u) {
        for (int i = 0; i < (int)dataOfUser[u].size(); ++i) {
            int id = dataOfUser[u][i];

            if (i < splitNum) {
                trainId.push_back(id);
            }
            else {
                validationId.push_back(id);
            }
        }
    }

    // pv(trainId.size());pv(validationId.size()); pn;pn; ////////////
    // for (int id : validationId) {
    //     pv(id);pv(userOfInput[id]);pn; //////
    // }
}

void loadTrainValidationData() {
    string path = "../../data/train/";

    int i = 0;

    vector<int> arrays[] = {trainId, validationId};

    for (int k = 0; k < 2; ++k) {
        vector<int>& ids = arrays[k];
        for (int id : ids) {
            ++i;
            if (i % 500 == 0) {
                cout << "loaded: "; pv(i);pn; ////////////////////////////////
            }

            stringstream ss;
            ss << id;
            string name;
            ss >> name;

            // pv(id);pv(name);pn; ///

            ifstream in(path + name + ".csv");
            string line;
            vector<Point> example;
            while (true) {
                getline(in, line);
                if (in.fail()) {
                    break;
                }

                ld x,y,z;
                sscanf(line.c_str(), "%Lf,%Lf,%Lf\n", &x, &y, &z);
                // pv(x);pv(y);pv(z);pn; ////
                Point p(x, y, z);
                example.push_back(p);
            }
            in.close();
            exampleDataWithId[id] = example;
        }
    }

    // for (Point p : exampleDataWithId[23999]) {
    //     pv(p.coord[0]);pv(p.coord[1]);pv(p.coord[2]);pn;///
    // }
}


void normalizeData(unordered_map<int, vector<Point>>& mp) {
    ld minCoord[3] = {1e18, 1e18, 1e18};
    ld maxCoord[3] = {-1e18, -1e18, -1e18};
    for (auto& pereche : mp) {
        // int id = pereche.first;
        auto& example = pereche.second;

        for (Point& p : example) {
            for (int k = 0; k < 3; ++k) {
                minCoord[k] = min(minCoord[k], p.coord[k]);
                maxCoord[k] = max(maxCoord[k], p.coord[k]);
            }
        }
    }

    // for (int k = 0; k < 3; ++k) {
    //     pv(k);pv(minCoord[k]);pv(maxCoord[k]);pn; ///
    // }

    for (auto& pereche : mp) {
        // int id = pereche.first;
        auto& example = pereche.second;

        for (Point& p : example) {
            for (int k = 0; k < 3; ++k) {
                p.coord[k] -= minCoord[k];
                p.coord[k] /= maxCoord[k] - minCoord[k];
            }
        }
    }

    // for (Point p : exampleDataWithId[23999]) {
    //     pv(p.coord[0]);pv(p.coord[1]);pv(p.coord[2]);pn;///
    // }
}

void trimData(unordered_map<int, vector<Point>>& mp) {
    const int increase = 1; ////////////////////////////////////////////////////////

    for (auto& pereche : mp) {
        int id = pereche.first;
        auto& example = pereche.second;

        vector<Point> newExample;
        for (int i = 0; i < (int)example.size(); i += increase) {
            Point p = example[i];
            newExample.push_back(p);

            // with average;
            // Point p = example[i];
            // for (int j = i + 1; j < i + increase; ++j) {
            //     for (int k = 0; k < 3; ++k) {
            //         p.coord[k] += example[j].coord[k];
            //     }
            // }

            // for (int k = 0; k < 3; ++k) {
            //     p.coord[k] /= increase;
            // }
            // newExample.push_back(p);


            // with extremes;
            // Point p = example[i];
            // for (int j = i + 1; j < i + increase; ++j) {
            //     if (i % increase % 2 == 0) {
            //         for (int k = 0; k < 3; ++k) {
            //             p.coord[k] = min(p.coord[k], example[j].coord[k]);
            //         }
            //     }
            //     else {
            //         for (int k = 0; k < 3; ++k) {
            //             p.coord[k] = max(p.coord[k], example[j].coord[k]);
            //         }
            //     }
            // }
            // newExample.push_back(p);


            // Point p = example[i];
            // for (int j = i + 1; j < i + increase; ++j) {
            //     for (int k = 0; k < 3; ++k) {
            //         for (int k = 0; k < 3; ++k) {
            //             p.coord[k] = min(p.coord[k], example[j].coord[k]);
            //         }
            //     }
            // }
            // newExample.push_back(p);

            // p = example[i];
            // for (int j = i + 1; j < i + increase; ++j) {
            //     for (int k = 0; k < 3; ++k) {
            //         for (int k = 0; k < 3; ++k) {
            //             p.coord[k] = max(p.coord[k], example[j].coord[k]);
            //         }
            //     }
            // }
            // newExample.push_back(p);
        }

        trimmedExampleDataWithId[id] = newExample;
    }

    // for (Point p : trimmedExampleDataWithId[15069]) {
    //     pv(p.coord[0]);pv(p.coord[1]);pv(p.coord[2]);pn;///
    // }
}

long double euclid(const Point& a, const Point& b) {
    ld ans = 0;
    for (int k = 0; k < 3; ++k) {
        ld val = a.coord[k] - b.coord[k];
        ans += val * val;
    }
    ans = sqrt(ans);

    return ans;
}

long double manhattan(const Point& a, const Point& b) {
    ld ans = 0;
    for (int k = 0; k < 3; ++k) {
        ld val = abs(a.coord[k] - b.coord[k]);
        ans += val;
    }

    return ans;
}

long double exampleDistanceDP(const vector<Point>& e1, const vector<Point>& e2, ld (*distance)(const Point&, const Point&)) {
    for (int i = 1; i <= (int)e1.size(); ++i) {
        for (int j = 1; j <= (int)e2.size(); ++j) {
            dp[i][j] = min( dp[i - 1][j], min(dp[i - 1][j - 1], dp[i][j - 1]) );
            dp[i][j] += distance(e1[i - 1], e2[j - 1]);
        }
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////
    // for (int i = 1; i <= (int)e1.size(); ++i) {
    //     for (int j = 1; j <= (int)e2.size(); ++j) {
    //         cout << dp[i][j] << ' ';
    //     }
    //     cout << '\n';
    // }
    // pn;pn;//

    return dp[ (int)e1.size() ][ (int)e2.size() ];
}


long double exampleDistanceLinear(const vector<Point>& e1, const vector<Point>& e2, ld (*distance)(const Point&, const Point&)) {
    ld ans = 0;
    for (int i = 1; i <= min((int)e1.size(), (int)e2.size()); ++i) {
        ans += distance(e1[i], e2[i]);
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////
    // for (int i = 1; i <= (int)e1.size(); ++i) {
    //     for (int j = 1; j <= (int)e2.size(); ++j) {
    //         cout << dp[i][j] << ' ';
    //     }
    //     cout << '\n';
    // }
    // pn;pn;//

    return ans;
}


int getClassOf(int hereId, int numNeighbours, ld (*distance)(const Point&, const Point&)) {
    vector<pair<int, ld>> sortedPairs;
    for (int thereId : trainId) {
        ld dist = exampleDistanceDP(trimmedExampleDataWithId[hereId], trimmedExampleDataWithId[thereId], distance);
        // ld dist = exampleDistanceLinear(trimmedExampleDataWithId[hereId], trimmedExampleDataWithId[thereId], distance);
        sortedPairs.push_back( {userOfInput[thereId], dist} );
    }

    sort(sortedPairs.begin(), sortedPairs.end(), [&](const pair<int, ld>& a, const pair<int, ld>& b) -> bool {
        return a.second < b.second;
    });


    // for (auto p : sortedPairs) {
    //     pv(p.first);pv(p.second); pn; ///////////
    // }
    // pn;pn;pn;pn; ////


    map<int, int> occ;
    int maxUser = -1, maxOcc = -1;
    for (int k = 0; k < numNeighbours; ++k) {
        int user = sortedPairs[k].first;
        occ[user] += 1;
        
        if (maxOcc < occ[user]) {
            maxOcc = occ[user];
            maxUser = user;
        }
    }

    return maxUser;
}

int globalCounter = 0, total = 0, correct = 0;
void getAllClasses(const vector<int>& targetIds, int numNeighbours, ld (*distance)(const Point&, const Point&), map<int, int>& predictedOfId) {
    for (int hereId : targetIds) {
        int predicted = getClassOf(hereId, numNeighbours, distance);
        predictedOfId[hereId] = predicted;

        globalCounter += 1;

        total += 1;
        if (predicted == userOfInput[hereId]) {
            correct += 1;
        }

        cout << "hereId: " << hereId << " is done. Predicted:" << predictedOfId[hereId] << "; Actual: " << userOfInput[hereId] << "\n"; ////
        ld accuracy = (ld) correct / (ld) total * 100;
        pv(correct);pv(total);pv(accuracy);pn;pn; //////
        pv(globalCounter);pn;pn;
    }
}

long double getAccuracy(map<int, int>& actual, map<int, int>& predicted) {
    return (ld) correct / (ld) total * 100;

    // int total = 0;
    // int correct = 0;
    // // assert(actual.size() == predicted.size()); ////////////////////////////////////////////////////////////

    // for (auto& it : predicted) {
    //     int id = it.first;
    //     int user = it.second;

    //     total += 1;
    //     if (user == actual[id]) {
    //         correct += 1;
    //     }

    //     pv(id);pv(actual[id]);pv(predicted[id]);pn; /////
    // }

    // pv(correct);pv(total);pn;pn; //////

    // return (ld)correct / (ld) total * 100;
}

int main() {
    cin.sync_with_stdio(false);
    cin.tie(0);

    // getTrainFiles(testFiles);
    getUserTestFiles(dataOfUser);

    buildMap(dataOfUser, userOfInput);

    splitTrainValidation(dataOfUser, trainId, validationId);

    loadTrainValidationData();
    normalizeData(exampleDataWithId);
    trimData(exampleDataWithId);


    // ld dist = exampleDistance(trimmedExampleDataWithId[15065], trimmedExampleDataWithId[23999], euclid);
    // ld dist = exampleDistance(trimmedExampleDataWithId[15069], trimmedExampleDataWithId[23999], euclid);


    {
        // id = 23995; user = 13; mai inseamna ceva?
        // int debugId[] = {10003,10005,10006,10007,10012,10013,10014,10016,10017,10018,10019,10020,10021,10022,10024};
        // for (int id : debugId) {
        //     int user = getClassOf(id, 1, euclid);
        //     pv(id);pv(user);pv(userOfInput[id]);pn; ///
        // }
    }

    trainId.resize(1 * trainId.size());
    // pv(trainId.size());
    // pv(validationId.size());

    //*
    // random_shuffle(validationId.begin(), validationId.end());
    // getAllClasses(validationId, 3, euclid, predictedUserOfInput);
    // getAllClasses(trainId, 1, euclid, predictedUserOfInput);
    // getAllClasses(vector<int>(validationId.begin(), validationId.begin() + 100), 3, euclid, predictedUserOfInput);
    getAllClasses(vector<int>(validationId.begin(), validationId.end()), 3, euclid, predictedUserOfInput);
    // getAllClasses(vector<int>(trainId.begin() + 900, trainId.begin() + 950), 1, manhattan, predictedUserOfInput);

    for (auto& it : predictedUserOfInput) {
        int id = it.first;
        int user = it.second;
        pv(id);pv(user);pv(userOfInput[id]);pn; ////////
    }

    // ld accuracy = getAccuracy(userOfInput, predictedUserOfInput);
    ld accuracy = (ld) correct / (ld) total * 100;
    pv(accuracy); cout << "%"; pn; //////////////////////////
    //*/
    
    return 0;
}